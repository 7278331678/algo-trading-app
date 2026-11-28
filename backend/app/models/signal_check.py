"""Planted-signal check: can the pipeline find an edge when we know one exists?

"No edge" on real data only means something if the same pipeline reliably finds an edge that
is really there. This check takes the REAL dataset (same stocks, days, features, costs and
backtester) and adds two extra features that carry a known amount of information about each
trade's real outcome:

    planted_<side> = strength * (2 * outcome - 1) + Normal(0, 1)

The outcome is the real triple-barrier result after costs, so a model that uses the feature
picks trades that really won on real prices, and the backtest measures that honestly.
In theory a perfect model reaches AUC = Phi(strength * sqrt(2)):

    strength 0.00 -> AUC 0.500  (null: pure noise; the pipeline must NOT find an edge)
    strength 0.05 -> AUC 0.528
    strength 0.09 -> AUC 0.551
    strength 0.20 -> AUC 0.611  (strong: the pipeline MUST find it and make money)

The planted features look into the future on purpose. They exist only inside this check and
are never saved as models.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from app.features import build

from . import dataset, study

log = logging.getLogger(__name__)

PLANTED = ["planted_long", "planted_short"]
DEFAULT_STRENGTHS = (0.0, 0.05, 0.09, 0.20)


def expected_auc(strength: float) -> float:
    return 0.5 * (1 + math.erf(strength))   # Phi(s * sqrt(2)) = 0.5 * (1 + erf(s))


def plant(df: pd.DataFrame, strength: float, seed: int = 11) -> pd.DataFrame:
    """Adds planted_long / planted_short in place (NaN where the outcome is unknown)."""
    rng = np.random.default_rng(seed)
    for side in ("long", "short"):
        y = df[f"t_{side}"].to_numpy(dtype=float)
        noise = rng.normal(0.0, 1.0, len(df))
        df[f"planted_{side}"] = np.where(np.isnan(y), np.nan, strength * (2 * y - 1) + noise).astype("float32")
    return df


def mean_auc(fold_metrics: list[dict], model: str = "ensemble") -> float | None:
    aucs = [m["auc"] for m in fold_metrics if m["model"] == model and m.get("auc") is not None]
    return round(float(np.mean(aucs)), 4) if aucs else None


def judge(strength: float, auc: float | None, verdict: str, holdout: dict | None, tol: float = 0.02) -> dict:
    """What each strength must show for the pipeline to be trusted."""
    exp = expected_auc(strength)
    if auc is None:
        return {"pass": False, "why": "no folds"}
    if strength == 0:
        ok = abs(auc - 0.5) <= tol and verdict != "edge_candidate"
        why = "noise stays at chance and is not called an edge" if ok else "found an edge in pure noise"
    elif exp >= 0.6:
        ok = auc >= exp - 0.03 and holdout is not None and (holdout.get("net_pnl") or 0) > 0
        why = "strong edge found and profitable after costs" if ok else "missed a strong, real edge"
    else:
        ok = auc >= exp - tol
        why = "weak edge detected (profit not required)" if ok else "missed a weak edge the models should see"
    return {"pass": bool(ok), "why": why}


def run(df: pd.DataFrame, clock, base, barrier: str, cost: dataset.CostModel, kinds: tuple[str, ...],
        decision_times: list | None, strengths=DEFAULT_STRENGTHS, max_train_rows: int = 600_000,
        seed: int = 11) -> dict:
    df = study.prepare(df, clock, barrier, cost)   # once: labels and targets are the same for every strength
    features = build.ALL_FEATURES + PLANTED
    rows = []
    for s in strengths:
        plant(df, s, seed)
        res = study.run_study(df, clock, base, barrier, cost, kinds, max_train_rows=max_train_rows,
                              decision_times=decision_times, features=features, prepared=True)
        auc = mean_auc(res["fold_metrics"])
        h = res["chosen_on_holdout"]
        row = {"strength": s, "expected_auc": round(expected_auc(s), 4), "measured_auc": auc,
               "auc_by_model": {k: mean_auc(res["fold_metrics"], k) for k in kinds},
               "verdict": res["verdict"],
               "holdout": {k: h.get(k) for k in ("model", "threshold", "trades", "net_pnl",
                                                  "gross_before_costs", "win_rate", "profit_factor")} if h else None,
               **judge(s, auc, res["verdict"], h)}
        log.info("planted strength %.2f: %s", s, row)
        rows.append(row)
    passed = all(r["pass"] for r in rows)
    # a lucky positive holdout on pure noise (strength 0) is possible and allowed; it is not an edge
    profitable = [r["strength"] for r in rows
                  if r["strength"] > 0 and r["holdout"] and (r["holdout"]["net_pnl"] or 0) > 0]
    return {"passed": passed, "results": rows,
            "weakest_profitable_strength": min(profitable) if profitable else None,
            "meaning": ("pipeline trusted: it ignores noise and finds real edges, so 'no edge' on real data is real"
                        if passed else "pipeline NOT trusted: fix this before believing any verdict")}
