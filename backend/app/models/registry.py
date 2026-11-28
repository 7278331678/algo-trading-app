"""Model registry: candidate -> shadow -> champion, decided only by results on unseen days.

Weekly, in this order:
1. Every shadow and the champion is scored on trading days AFTER the day it was trained up to
   (days it has never seen), through the same backtester and costs. No orders are placed.
2. A shadow with >= MIN_SHADOW_DAYS such days is decided once: promoted if its forward results
   pass and beat the champion over the same days, otherwise retired.
3. The champion is retired if its recent forward results go bad. Having no champion is a valid,
   safe state: Phase 6 then trades nothing from models.
4. The new candidate trained this week becomes a shadow only if its holdout verdict passes the gate;
   otherwise it is marked rejected and kept for the record.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import date, time

import joblib
import numpy as np
import pandas as pd

from app.backtest import runner
from app.backtest.engine import BacktestConfig

from . import dataset, regime, study

log = logging.getLogger(__name__)

GATE_VERDICTS = {"edge_candidate"}   # weak_positive is not enough to risk shadow slots
MIN_SHADOW_DAYS = 20                 # about four weeks of unseen trading days
MIN_SHADOW_TRADES = 15
MIN_PROFIT_FACTOR = 1.2
MAX_SHADOWS = 2
CHAMPION_REVIEW_DAYS = 20
CHAMPION_MIN_PF = 0.8


# ------------------------------------------------------------------ pure decisions (unit tested)
def gate(verdict: str) -> tuple[str, str]:
    if verdict in GATE_VERDICTS:
        return "shadow", f"holdout verdict {verdict}"
    return "rejected", f"holdout verdict {verdict} (needs {'/'.join(sorted(GATE_VERDICTS))})"


def forward_ok(s: dict) -> tuple[bool, str]:
    trades, pnl, pf = s.get("trades", 0) or 0, s.get("net_pnl", 0) or 0, s.get("profit_factor") or 0
    if trades < MIN_SHADOW_TRADES:
        return False, f"only {trades} trades (needs {MIN_SHADOW_TRADES})"
    if s.get("would_halt_on"):
        return False, f"kill switch would have tripped on {s['would_halt_on']}"
    if pnl <= 0:
        return False, f"net P&L {pnl:.0f} after costs"
    if pf < MIN_PROFIT_FACTOR:
        return False, f"profit factor {pf:.2f} (needs {MIN_PROFIT_FACTOR})"
    return True, f"{trades} trades, net {pnl:.0f}, PF {pf:.2f}"


def decide_shadow(summary: dict, trading_days: int, champion_same_window: dict | None) -> tuple[str, str]:
    """-> ('wait'|'promote'|'retire', reason)"""
    if trading_days < MIN_SHADOW_DAYS:
        return "wait", f"{trading_days}/{MIN_SHADOW_DAYS} unseen trading days so far"
    ok, why = forward_ok(summary)
    if not ok:
        return "retire", f"failed shadow: {why}"
    if champion_same_window is not None and (champion_same_window.get("net_pnl") or 0) >= summary["net_pnl"]:
        return "retire", f"passed ({why}) but did not beat the champion over the same days"
    return "promote", f"passed shadow: {why}"


def decide_champion(recent: dict) -> tuple[str, str]:
    """-> ('keep'|'retire', reason), judged on the last CHAMPION_REVIEW_DAYS unseen days."""
    if recent.get("would_halt_on"):
        return "retire", f"kill switch would have tripped on {recent['would_halt_on']}"
    if (recent.get("trades") or 0) >= 10 and (recent.get("profit_factor") or 0) < CHAMPION_MIN_PF:
        return "retire", f"profit factor {recent['profit_factor']:.2f} over the last {CHAMPION_REVIEW_DAYS} days"
    return "keep", "forward results acceptable"


# ------------------------------------------------------------------ scoring on unseen days
def score(df: pd.DataFrame, models: dict, features: list[str], kinds,
          decision_times: list[time] | None) -> pd.DataFrame:
    """Adds p_<side>_<kind> and ensemble p_<side> columns (NaN off decision times)."""
    out = df
    rows = out["decision_time"].dt.time.isin(set(decision_times)).to_numpy() if decision_times \
        else np.ones(len(out), bool)
    X = out.loc[rows, features].to_numpy(dtype=np.float32)
    for side in ("long", "short"):
        for kind in kinds:
            col = np.full(len(out), np.nan)
            if len(X):
                col[rows] = models[f"{side}_{kind}"].predict(X)
            out[f"p_{side}_{kind}"] = col
        out[f"p_{side}"] = out[[f"p_{side}_{k}" for k in kinds]].mean(axis=1)
    return out


def forward_eval(df: pd.DataFrame, clock, base: BacktestConfig, models: dict, manifest: dict, chosen: dict,
                 start_after: date, end: date | None = None) -> tuple[dict, int, date | None, date | None]:
    """Backtest a saved model on days after `start_after`. Returns (summary, trading_days, first, last)."""
    keep = df["day"] > start_after
    if end:
        keep &= df["day"] <= end
    part = df[keep]
    if part.empty:
        return {"trades": 0}, 0, None, None
    spec = manifest.get("spec", {})
    dts = [time.fromisoformat(t) for t in spec.get("decision_times") or []] or None
    kinds = tuple(spec.get("kinds") or manifest.get("kinds") or ("hgb", "logit"))
    cost = dataset.CostModel(slippage_bps_per_side=float(spec.get("slippage_bps", 3.0)))
    missing = [f for f in manifest["features"] if f not in part.columns]
    if missing:
        raise RuntimeError(f"model needs features this run did not build: {missing[:5]}")
    part = runner.add_exit_times(part, clock)
    part = score(part, models, manifest["features"], kinds, dts)
    part["bad_day"] = regime.bad_day_flag(part)
    cfg = study.backtest_config(base, manifest["barrier"], cost)
    s = study._bt(part, cfg, chosen["model"], float(chosen["threshold"]))["summary"]
    n_days = int(part["day"].nunique())
    return s, n_days, min(part["day"]), max(part["day"])


def load_models(get_bytes, prefix: str) -> tuple[dict, dict]:
    manifest = json.loads(get_bytes(f"{prefix}/manifest.json"))
    models = {name: joblib.load(io.BytesIO(get_bytes(f"{prefix}/{name}.joblib"))) for name in manifest["models"]}
    return manifest, models


# ------------------------------------------------------------------ database side
def _set_status(conn, pk: int, status: str, reason: str) -> None:
    from sqlalchemy import text

    conn.execute(text("UPDATE model_runs SET status = :s, status_reason = :r, status_changed_at = now()"
                      " WHERE id = :i"), {"s": status, "r": reason, "i": pk})
    log.info("model run %s -> %s: %s", pk, status, reason)


def _record_eval(conn, pk: int, first, last, n_days: int, summary: dict) -> None:
    from sqlalchemy import text

    conn.execute(text("INSERT INTO model_evals (model_run_id, from_day, to_day, trading_days, summary)"
                      " VALUES (:i, :f, :t, :n, CAST(:s AS jsonb))"),
                 {"i": pk, "f": first, "t": last, "n": n_days, "s": json.dumps(summary, default=str)})


def weekly_review(engine, get_bytes, df: pd.DataFrame, clock, base: BacktestConfig) -> dict:
    """Steps 1-3 of the module docstring. Returns what was decided."""
    from sqlalchemy import text

    with engine.connect() as conn:
        live = conn.execute(text("SELECT id, status, artifacts, chosen, trained_until FROM model_runs"
                                 " WHERE status IN ('shadow', 'champion') ORDER BY created_at")).mappings().all()
    champ = next((r for r in live if r["status"] == "champion"), None)
    loaded = {r["id"]: load_models(get_bytes, r["artifacts"]) for r in live}
    decisions = []
    latest = max(df["day"])

    with engine.begin() as conn:
        promoted = None
        passing = []
        for r in [r for r in live if r["status"] == "shadow"]:
            manifest, models = loaded[r["id"]]
            s, n, first, last = forward_eval(df, clock, base, models, manifest, r["chosen"], r["trained_until"])
            if n:
                _record_eval(conn, r["id"], first, last, n, s)
            champ_same = None
            if champ and n >= MIN_SHADOW_DAYS:
                cm, cmod = loaded[champ["id"]]
                champ_same = forward_eval(df, clock, base, cmod, cm, champ["chosen"],
                                          max(r["trained_until"], champ["trained_until"]))[0]
            action, why = decide_shadow(s, n, champ_same)
            decisions.append({"id": r["id"], "status": "shadow", "action": action, "why": why, "forward": s})
            if action == "retire":
                _set_status(conn, r["id"], "retired", why)
            elif action == "promote":
                passing.append((r["id"], s["net_pnl"], why))
        best = max(passing, key=lambda p: p[1]) if passing else None
        for other in passing:
            if other is not best:
                _set_status(conn, other[0], "retired", "another shadow promoted with better forward results")
        if best:
            if champ:
                _set_status(conn, champ["id"], "retired", f"replaced by model run {best[0]}")
            _set_status(conn, best[0], "champion", best[2])
            promoted = best[0]
        elif champ:
            manifest, models = loaded[champ["id"]]
            window_start = _nth_last_day(df, CHAMPION_REVIEW_DAYS)
            s, n, first, last = forward_eval(df, clock, base, models, manifest, champ["chosen"],
                                             max(champ["trained_until"], window_start))
            if n:
                _record_eval(conn, champ["id"], first, last, n, s)
            action, why = decide_champion(s)
            decisions.append({"id": champ["id"], "status": "champion", "action": action, "why": why, "forward": s})
            if action == "retire":
                _set_status(conn, champ["id"], "retired", why)
    return {"latest_day": str(latest), "promoted": promoted, "decisions": decisions}


def _nth_last_day(df: pd.DataFrame, n: int) -> date:
    days = sorted(df["day"].unique())
    return days[-n - 1] if len(days) > n else days[0]


def register_candidate(engine, pk: int, verdict: str) -> tuple[str, str]:
    """Step 4: gate the new run, then keep at most MAX_SHADOWS shadows (oldest retired)."""
    from sqlalchemy import text

    status, why = gate(verdict)
    with engine.begin() as conn:
        _set_status(conn, pk, status, why)
        if status == "shadow":
            ids = [r[0] for r in conn.execute(text(
                "SELECT id FROM model_runs WHERE status = 'shadow' ORDER BY created_at DESC"))]
            for old in ids[MAX_SHADOWS:]:
                _set_status(conn, old, "retired", f"shadow slots full (max {MAX_SHADOWS}); newer runs kept")
    return status, why
