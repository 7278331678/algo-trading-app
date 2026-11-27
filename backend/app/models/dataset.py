"""Training data: features + labels for a chosen barrier, with cost-aware targets.

A trade only counts as a win (target = 1) if its return beats the round-trip cost estimate,
so the models learn to find trades that pay for themselves, not just ones that hit a target.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.features import build, labels

# barrier presets: (unit, multiplier of that unit for one barrier "R", target in R, stop in R)
BARRIERS = {
    "atr5_1.0x1.5": ("atr5", 1.0, 1.5, 1.0),
    "daily_0.4x0.6": ("daily", 0.4, 1.5, 1.0),   # stop 0.4 x daily ATR, target 0.6 x daily ATR
    "daily_0.6x1.2": ("daily", 0.6, 2.0, 1.0),   # stop 0.6 x daily ATR, target 1.2 x daily ATR (bigger moves)
}

META = ["ts", "decision_time", "day", "symbol", "sector_index", "open", "high", "low", "close",
        "atr14", "daily_atr_pct", "exclude", "in_universe"]


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost as a fraction of trade value: Zerodha charges at ~Rs 1 lakh per trade
    (~0.082%) plus slippage on entry and exit."""
    charges_pct: float = 0.00082
    slippage_bps_per_side: float = 3.0

    @property
    def round_trip(self) -> float:
        return self.charges_pct + 2 * self.slippage_bps_per_side / 10_000


def load(list_keys, get_bytes, symbols: set[str] | None = None) -> pd.DataFrame:
    prefix = f"features/{build.FEATURE_VERSION}/5minute/"
    cols = list(dict.fromkeys(META + build.ALL_FEATURES))  # some columns are both meta and feature
    frames = []
    for key in sorted(list_keys(prefix)):
        sym = key.rsplit("/", 1)[-1].removesuffix(".parquet")
        if symbols and sym not in symbols:
            continue
        df = pd.read_parquet(io.BytesIO(get_bytes(key)), columns=cols)
        for c in build.ALL_FEATURES + ["atr14", "daily_atr_pct"]:
            df[c] = df[c].astype("float32")
        df["day"] = pd.to_datetime(df["day"]).dt.date
        frames.append(df)
    if not frames:
        raise RuntimeError("no feature files found: run build-features first")
    return pd.concat(frames, ignore_index=True)


def relabel(df: pd.DataFrame, barrier: str) -> pd.DataFrame:
    """Recompute triple-barrier labels for `barrier` (needs exit_time from runner.add_exit_times)."""
    basis, k, tm, sm = BARRIERS[barrier]
    out = df  # modified in place to keep memory down (millions of rows)
    out["unit"] = out["atr14"].astype(float) * k if basis == "atr5" else \
        out["daily_atr_pct"].astype(float) * out["close"].astype(float) * k
    parts = []
    for _, g in out.groupby("symbol", sort=False):
        g = g.sort_values("ts")
        ex = dict(zip(g["day"], g["exit_time"], strict=False))
        parts.append(labels.triple_barrier(g, ex, target_mult=tm, stop_mult=sm, unit_col="unit"))
    lab = pd.concat(parts).reindex(out.index)
    for c in labels.LABEL_COLUMNS:
        out[c] = lab[c]
    return out


def add_targets(df: pd.DataFrame, cost: CostModel) -> pd.DataFrame:
    """y_* = 1 only if the trade's return beats the round-trip cost (in place)."""
    out = df
    c = cost.round_trip
    out["t_long"] = np.where(out["ret_long"].notna(), (out["ret_long"] > c).astype(float), np.nan)
    out["t_short"] = np.where(out["ret_short"].notna(), (out["ret_short"] > c).astype(float), np.nan)
    return out


def trainable(df: pd.DataFrame) -> pd.Series:
    """Rows a model may learn from: labelled, not excluded, stock in the index that day."""
    return df["ret_long"].notna() & ~df["exclude"].astype(bool) & df["in_universe"].astype(bool)
