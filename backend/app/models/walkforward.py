"""Walk-forward training by calendar quarter.

For every test quarter the models are trained only on data that ended before the quarter
started, so every prediction is out-of-sample, as it would have been live. Within each training
window the most recent 15% (by time) is held back to calibrate probabilities (isotonic), so a
predicted 0.60 means "about 60% of such trades beat their costs".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

SIDES = ("long", "short")


def make_model(kind: str, seed: int = 7, n_rows: int = 1_000_000):
    if kind == "hgb":   # histogram gradient boosting (the LightGBM algorithm family), handles NaN natively
        leaf = int(min(500, max(50, n_rows // 200)))   # smaller datasets (one decision a day) need smaller leaves
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=leaf, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    if kind == "logit":  # simple, robust baseline
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=500))
    raise ValueError(f"unknown model kind {kind}")


@dataclass
class CalibratedModel:
    kind: str
    model: object
    iso: IsotonicRegression | None

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        return self.iso.predict(raw) if self.iso is not None else raw


def fit_calibrated(kind: str, X: np.ndarray, y: np.ndarray, ts: np.ndarray, max_rows: int, seed: int = 7,
                   calib_frac: float = 0.15) -> CalibratedModel:
    order = np.argsort(ts, kind="stable")
    cut = int(len(order) * (1 - calib_frac))
    fit_idx, cal_idx = order[:cut], order[cut:]
    rng = np.random.default_rng(seed)
    if len(fit_idx) > max_rows:
        fit_idx = np.sort(rng.choice(fit_idx, max_rows, replace=False))
    if len(cal_idx) > max_rows // 3:
        cal_idx = np.sort(rng.choice(cal_idx, max_rows // 3, replace=False))
    model = make_model(kind, seed, len(fit_idx))
    model.fit(X[fit_idx], y[fit_idx])
    iso = None
    if len(np.unique(y[cal_idx])) == 2:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(model.predict_proba(X[cal_idx])[:, 1], y[cal_idx])
    return CalibratedModel(kind, model, iso)


def quarter_folds(days: pd.Series, min_train_quarters: int = 2) -> list[pd.Period]:
    q = pd.PeriodIndex(pd.to_datetime(days), freq="Q")
    quarters = sorted(set(q))
    return quarters[min_train_quarters:]


def fold_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ok = ~np.isnan(y)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auc": None}
    top = p >= np.quantile(p, 0.9)
    return {"n": int(len(y)), "base_rate": round(float(y.mean()), 4), "mean_pred": round(float(p.mean()), 4),
            "auc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 5),
            "top_decile_rate": round(float(y[top].mean()), 4),
            "top_decile_lift": round(float(y[top].mean() / max(y.mean(), 1e-9)), 3)}


def run_walkforward(df: pd.DataFrame, features: list[str], train_mask: pd.Series,
                    kinds: tuple[str, ...] = ("hgb", "logit"), max_train_rows: int = 600_000,
                    min_train_quarters: int = 2, seed: int = 7,
                    min_train_rows: int = 10_000) -> tuple[pd.DataFrame, list[dict]]:
    """Returns out-of-sample predictions (rows of the test quarters) and per-fold metrics."""
    quarter = pd.PeriodIndex(pd.to_datetime(df["day"]), freq="Q")
    X_all = df[features].to_numpy(dtype=np.float32)
    ts_all = df["ts"].to_numpy()
    preds = []
    metrics = []
    for q in quarter_folds(df["day"], min_train_quarters):
        train = (quarter < q) & train_mask.to_numpy()
        test = (quarter == q) & df["ret_long"].notna().to_numpy()
        if train.sum() < min_train_rows or test.sum() == 0:
            continue
        out = pd.DataFrame(index=df.index[test])
        for side in SIDES:
            y_tr = df.loc[train, f"t_{side}"].to_numpy(dtype=float)
            for kind in kinds:
                m = fit_calibrated(kind, X_all[train], y_tr, ts_all[train], max_train_rows, seed)
                p = m.predict(X_all[test])
                out[f"p_{side}_{kind}"] = p
                eval_mask = train_mask.to_numpy()[test]
                fm = fold_metrics(df.loc[test, f"t_{side}"].to_numpy(dtype=float)[eval_mask], p[eval_mask])
                metrics.append({"quarter": str(q), "side": side, "model": kind, **fm})
                log.info("fold %s %s %s: %s", q, side, kind, fm)
            out[f"p_{side}"] = out[[f"p_{side}_{k}" for k in kinds]].mean(axis=1)
            fm = fold_metrics(df.loc[test, f"t_{side}"].to_numpy(dtype=float)[eval_mask],
                              out[f"p_{side}"].to_numpy()[eval_mask])
            metrics.append({"quarter": str(q), "side": side, "model": "ensemble", **fm})
        preds.append(out)
    if not preds:
        raise RuntimeError("not enough history for walk-forward folds")
    return pd.concat(preds), metrics


def fit_final(df: pd.DataFrame, features: list[str], train_mask: pd.Series,
              kinds: tuple[str, ...] = ("hgb", "logit"), max_train_rows: int = 600_000,
              seed: int = 7) -> dict[str, CalibratedModel]:
    """Models trained on all history, for use from the next trading day."""
    X = df.loc[train_mask, features].to_numpy(dtype=np.float32)
    ts = df.loc[train_mask, "ts"].to_numpy()
    out = {}
    for side in SIDES:
        y = df.loc[train_mask, f"t_{side}"].to_numpy(dtype=float)
        for kind in kinds:
            out[f"{side}_{kind}"] = fit_calibrated(kind, X, y, ts, max_train_rows, seed)
    return out
