"""Phase 4 study: walk-forward models -> threshold chosen on early quarters -> tested on later ones.

Honesty rules built in:
- every prediction comes from a model trained only on earlier quarters
- the confidence threshold (and which model) is chosen on the FIRST half of the out-of-sample
  quarters, then judged on the SECOND half, which played no part in any choice
- the backtest uses the same barriers and slippage the labels were built with
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, replace

import joblib
import numpy as np
import pandas as pd

from app.backtest import metrics as bt_metrics
from app.backtest import runner, strategies
from app.backtest.engine import BacktestConfig, run_backtest
from app.features import build
from app.features.pipeline import code_hash

from . import dataset, regime, walkforward

log = logging.getLogger(__name__)

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MIN_TRADES_TO_SELECT = 30


def backtest_config(base: BacktestConfig, barrier: str, cost: dataset.CostModel) -> BacktestConfig:
    basis, k, tm, sm = dataset.BARRIERS[barrier]
    return replace(base, stop_basis="atr5" if basis == "atr5" else "daily", stop_mult=k * sm,
                   target_mult=k * tm, slippage_bps=cost.slippage_bps_per_side, max_drawdown_halt_pct=1e9)


def _bt(frame: pd.DataFrame, cfg: BacktestConfig, model: str, thr: float) -> dict:
    sig = strategies.model_strategy(frame, thr, model=model)
    res = run_backtest(sig, cfg)
    s = bt_metrics.summarize(res.trades, res.daily, cfg.capital)
    s["would_halt_on"] = bt_metrics.halt_day(res.daily, 15.0)
    s["profitable_quarters"] = bt_metrics.consistency(bt_metrics.by_period(res.trades, res.daily, cfg.capital))
    return {"model": model, "threshold": thr, "summary": s, "trades": res.trades, "daily": res.daily}


def prepare(df: pd.DataFrame, clock, barrier: str, cost: dataset.CostModel) -> pd.DataFrame:
    """Exit times, labels for `barrier` and cost-aware targets. Returns a new frame."""
    df = runner.add_exit_times(df, clock)
    dataset.relabel(df, barrier)
    dataset.add_targets(df, cost)
    return df


def run_study(df: pd.DataFrame, clock, base: BacktestConfig, barrier: str = "daily_0.4x0.6",
              cost: dataset.CostModel | None = None, kinds: tuple[str, ...] = ("hgb", "logit"),
              thresholds: list[float] | None = None, max_train_rows: int = 600_000,
              decision_times: list | None = None, features: list[str] | None = None,
              prepared: bool = False) -> dict:
    """decision_times: if given (e.g. [time(9, 45)]), models train and trade only on candles closing at
    those times: one decision per stock per day instead of one per 5-minute candle.
    prepared: df already went through prepare() with this barrier and cost (lets callers reuse it).
    features: model inputs (default: build.ALL_FEATURES)."""
    cost = cost or dataset.CostModel()
    thresholds = thresholds or THRESHOLDS
    features = features or build.ALL_FEATURES
    if not prepared:
        df = prepare(df, clock, barrier, cost)
    mask = dataset.trainable(df)
    log.info("dataset: %d rows, %d trainable, long win rate after costs %.3f", len(df), int(mask.sum()),
             float(df.loc[mask, "t_long"].mean()))

    final_mask = mask
    if decision_times:
        at = df["decision_time"].dt.time.isin(set(decision_times))
        final_mask = mask & at   # the model used live must be trained on the same kind of rows
        sub = df[at]
        preds, fold_metrics = walkforward.run_walkforward(sub, features, mask[at], kinds, max_train_rows,
                                                          min_train_rows=300)
        # backtests need every candle of the test quarters (to manage trades); predictions only at decision times
        pq = set(pd.PeriodIndex(pd.to_datetime(df.loc[preds.index, "day"]), freq="Q"))
        in_test = pd.PeriodIndex(pd.to_datetime(df["day"]), freq="Q").isin(pq)
        test = df[in_test].join(preds)
    else:
        preds, fold_metrics = walkforward.run_walkforward(df, features, mask, kinds, max_train_rows)
        test = df.loc[preds.index].join(preds)
    test["bad_day"] = regime.bad_day_flag(test)
    # backtests only need these columns (avoids copying ~60 columns x millions of rows per run)
    keep = ["ts", "day", "symbol", "open", "high", "low", "close", "atr14", "daily_atr_pct", "exit_time",
            "exclude", "in_universe", "rel_strength", "bad_day", *preds.columns]
    test = test[keep]

    event_rows = _event_row_metrics(df, preds, mask, kinds)
    quarters = sorted(pd.PeriodIndex(pd.to_datetime(test["day"]), freq="Q").unique())
    half = max(1, len(quarters) // 2)
    sel_q, hold_q = set(quarters[:half]), set(quarters[half:])
    tq = pd.PeriodIndex(pd.to_datetime(test["day"]), freq="Q")
    selection, holdout = test[tq.isin(sel_q)], test[tq.isin(hold_q)]
    cfg = backtest_config(base, barrier, cost)

    models = list(kinds) + ["ensemble"]
    sel_results = [_bt(selection, cfg, m, t) for m in models for t in thresholds]
    eligible = [r for r in sel_results if r["summary"].get("trades", 0) >= MIN_TRADES_TO_SELECT]
    chosen = max(eligible, key=lambda r: r["summary"]["net_pnl"]) if eligible else None
    hold_results = [_bt(holdout, cfg, m, t) for m in models for t in thresholds] if len(hold_q) else []
    chosen_hold = next((r for r in hold_results if chosen and r["model"] == chosen["model"]
                        and r["threshold"] == chosen["threshold"]), None)

    def brief(r):
        s = r["summary"]
        return {"model": r["model"], "threshold": r["threshold"],
                **{k: s.get(k) for k in ("trades", "trades_per_day", "net_pnl", "gross_before_costs",
                                         "cost_per_trade", "win_rate", "profit_factor", "max_drawdown_pct",
                                         "profitable_quarters", "would_halt_on")}}

    verdict = "no_edge"
    if chosen_hold and chosen_hold["summary"].get("trades", 0) >= MIN_TRADES_TO_SELECT:
        s = chosen_hold["summary"]
        if s["net_pnl"] > 0 and (s.get("profit_factor") or 0) >= 1.3 and not s.get("would_halt_on"):
            verdict = "edge_candidate"
        elif s["net_pnl"] > 0:
            verdict = "weak_positive"
    return {
        "barrier": barrier, "cost_round_trip_pct": round(cost.round_trip * 100, 4),
        "decision_times": [str(t) for t in decision_times] if decision_times else "every candle",
        "rows": int(len(df)), "trainable_rows": int(mask.sum()),
        "base_win_rate_after_costs": {s: round(float(df.loc[mask, f"t_{s}"].mean()), 4) for s in ("long", "short")},
        "selection_quarters": [str(q) for q in sorted(sel_q)], "holdout_quarters": [str(q) for q in sorted(hold_q)],
        "fold_metrics": fold_metrics,
        "event_rows": event_rows,
        "selection": [brief(r) for r in sel_results],
        "chosen": brief(chosen) if chosen else None,
        "holdout": [brief(r) for r in hold_results],
        "chosen_on_holdout": brief(chosen_hold) if chosen_hold else None,
        "verdict": verdict,
        "_frames": {"predictions": test.loc[test[preds.columns[0]].notna(), ["symbol", "ts", *preds.columns]],
                    "chosen_holdout": chosen_hold},
        "_df": df, "_mask": mask, "_config": cfg, "_features": features,
        "_final_mask": final_mask,
    }


def _event_row_metrics(df: pd.DataFrame, preds: pd.DataFrame, mask: pd.Series, kinds) -> dict | None:
    """Out-of-sample AUC on decisions that had material news since the last close (where news should matter)."""
    if "ev_material_since_close" not in df:
        return None
    rows = preds.index
    ok = mask.loc[rows].to_numpy() & (df.loc[rows, "ev_material_since_close"].to_numpy() > 0)
    out = {"rows": int(ok.sum())}
    for side in walkforward.SIDES:
        y = df.loc[rows, f"t_{side}"].to_numpy(dtype=float)[ok]
        for m in [*kinds, "ensemble"]:
            col = f"p_{side}" if m == "ensemble" else f"p_{side}_{m}"
            out[f"{side}/{m}"] = walkforward.fold_metrics(y, preds[col].to_numpy()[ok]).get("auc")
    return out


def trained_until(study: dict):
    """Last trading day the final models saw. Forward (shadow) evaluation starts after it."""
    df, m = study["_df"], study["_final_mask"]
    return max(df.loc[m, "day"])


def save(engine, put_bytes, run_id: str, study: dict, final_models: dict, kinds, spec: dict | None = None) -> int:
    from sqlalchemy import text

    spec = spec or {}
    prefix = f"models/{run_id}"
    buf = io.BytesIO()
    study["_frames"]["predictions"].to_parquet(buf, index=False)
    put_bytes(f"{prefix}/predictions.parquet", buf.getvalue())
    for name, model in final_models.items():
        b = io.BytesIO()
        joblib.dump(model, b)
        put_bytes(f"{prefix}/{name}.joblib", b.getvalue())
    public = {k: v for k, v in study.items() if not k.startswith("_")}
    until = trained_until(study)
    manifest = {"run_id": run_id, "features": study.get("_features", build.ALL_FEATURES),
                "feature_version": build.FEATURE_VERSION, "spec": spec, "trained_until": str(until),
                "code_hash": code_hash(), "models": sorted(final_models), "kinds": list(kinds),
                "backtest_config": {k: str(v) for k, v in asdict(study["_config"]).items()}, **public}
    put_bytes(f"{prefix}/manifest.json", json.dumps(manifest, indent=2, default=str).encode())
    with engine.begin() as conn:
        return conn.execute(text(
            "INSERT INTO model_runs (run_id, barrier, verdict, chosen, chosen_on_holdout, summary, artifacts,"
            " spec, trained_until)"
            " VALUES (:r, :b, :v, CAST(:c AS jsonb), CAST(:h AS jsonb), CAST(:s AS jsonb), :a,"
            " CAST(:sp AS jsonb), :tu) RETURNING id"),
            {"r": run_id, "b": study["barrier"], "v": study["verdict"],
             "c": json.dumps(study["chosen"], default=str), "h": json.dumps(study["chosen_on_holdout"], default=str),
             "s": json.dumps(public, default=str), "a": prefix, "sp": json.dumps(spec, default=str),
             "tu": until}).scalar_one()


def nan_to_none(x):
    return None if isinstance(x, float) and np.isnan(x) else x
