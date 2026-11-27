"""Performance metrics for a backtest, overall and per period (walk-forward style report)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, daily: pd.DataFrame, capital: float, halted_on=None) -> dict:
    if trades.empty:
        return {"trades": 0, "net_pnl": 0.0, "note": "no trades", "halted_on": str(halted_on) if halted_on else None}
    slip = float(trades["slippage"].sum()) if "slippage" in trades else 0.0
    wins = trades[trades["net"] > 0]
    losses = trades[trades["net"] <= 0]
    gross_profit, gross_loss = wins["net"].sum(), -losses["net"].sum()
    eq = daily["equity"]
    peak = eq.cummax()
    dd = (eq - peak) / peak
    rets = daily["net"] / capital
    sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(250)) if rets.std(ddof=1) > 0 else float("nan")
    return {
        "trades": int(len(trades)),
        "days": int(len(daily)),
        "trades_per_day": round(len(trades) / max(len(daily), 1), 2),
        "win_rate": round(float((trades["net"] > 0).mean()), 4),
        "gross_pnl": round(float(trades["gross"].sum()), 2),
        "gross_before_costs": round(float(trades["gross"].sum()) + slip, 2),
        "slippage": round(slip, 2),
        "charges": round(float(trades["charges"].sum()), 2),
        "cost_per_trade": round((slip + float(trades["charges"].sum())) / len(trades), 2),
        "halted_on": str(halted_on) if halted_on else None,
        "net_pnl": round(float(trades["net"].sum()), 2),
        "return_pct": round(float(trades["net"].sum() / capital * 100), 2),
        "profit_factor": round(float(gross_profit / gross_loss), 3) if gross_loss > 0 else float("inf"),
        "avg_r": round(float(trades["r_multiple"].mean()), 4),
        "expectancy_per_trade": round(float(trades["net"].mean()), 2),
        "charges_share_of_gross_abs": round(float(trades["charges"].sum() / max(trades["gross"].abs().sum(), 1e-9)), 4),
        "max_drawdown_pct": round(float(dd.min() * 100), 2),
        "sharpe_daily_annualised": round(sharpe, 3) if np.isfinite(sharpe) else None,
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        "long_share": round(float((trades["side"] == "long").mean()), 3),
    }


def by_period(trades: pd.DataFrame, daily: pd.DataFrame, capital: float, freq: str = "Q") -> list[dict]:
    """Per-quarter results: how consistent the strategy is over time."""
    if trades.empty:
        return []
    t = trades.copy()
    t["period"] = pd.PeriodIndex(pd.to_datetime(t["day"]), freq=freq).astype(str)
    d = daily.copy()
    d["period"] = pd.PeriodIndex(pd.to_datetime(d["day"]), freq=freq).astype(str)
    out = []
    for p, tp in t.groupby("period"):
        dp = d[d["period"] == p]
        gp, gl = tp.loc[tp["net"] > 0, "net"].sum(), -tp.loc[tp["net"] <= 0, "net"].sum()
        out.append({"period": p, "trades": int(len(tp)), "net_pnl": round(float(tp["net"].sum()), 2),
                    "win_rate": round(float((tp["net"] > 0).mean()), 4),
                    "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
                    "days": int(len(dp))})
    return out


def consistency(periods: list[dict]) -> float | None:
    if not periods:
        return None
    return round(sum(1 for p in periods if p["net_pnl"] > 0) / len(periods), 3)


def halt_day(daily: pd.DataFrame, halt_pct: float) -> str | None:
    """First day the kill switch would stop trading (equity halt_pct below its running peak)."""
    if daily.empty:
        return None
    eq = daily["equity"].to_numpy()
    peak = np.maximum.accumulate(np.r_[daily["equity"].iloc[0] - daily["net"].iloc[0], eq])[:-1]
    hit = np.nonzero(eq <= peak * (1 - halt_pct / 100))[0]
    return str(daily["day"].iloc[hit[0] + 1]) if len(hit) and hit[0] + 1 < len(daily) else None
