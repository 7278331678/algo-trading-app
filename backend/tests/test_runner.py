import pickle
from datetime import time

import numpy as np
import pandas as pd

from app.backtest import runner, strategies
from app.backtest.engine import BacktestConfig
from app.core.market_clock import MarketClock


def synthetic(symbol, bars, rng):
    days = pd.bdate_range("2026-11-02", periods=10)
    rows = []
    price = 500.0
    for d in days:
        for k in range(bars):
            o = price
            c = o * (1 + rng.normal(0, 0.002))
            rows.append({"ts": (d + pd.Timedelta(hours=9, minutes=15 + 5 * k)).tz_localize("Asia/Kolkata"),
                         "day": d, "symbol": symbol, "open": o, "high": max(o, c) * 1.001,
                         "low": min(o, c) * 0.999, "close": c, "atr14": c * 0.002, "daily_atr_pct": 0.015,
                         "exclude": False, "in_universe": True, "rvol": 2.0, "adx14": 30.0,
                         "or_breakout": 1.0 if k > 3 else np.nan, "vwap_dist": 0.01, "ema50_15m_dist": 0.01,
                         "st_dir": 1.0, "nifty_ret_open": 0.0})
            price = c
    return pd.DataFrame(rows)


def test_load_exit_times_and_study(monkeypatch):
    rng = np.random.default_rng(3)
    store = {
        "features/v1/5minute/CASCO.parquet": pickle.dumps(synthetic("CASCO", 72, rng)),   # stops 15:10
        "features/v1/5minute/NONCAS.parquet": pickle.dumps(synthetic("NONCAS", 75, rng)), # runs to 15:25
    }
    monkeypatch.setattr(pd, "read_parquet", lambda buf, columns=None: pickle.loads(buf.getvalue())[columns])
    df = runner.load_features(lambda p: [k for k in store if k.startswith(p)], store.__getitem__, None, None)
    assert set(df["symbol"]) == {"CASCO", "NONCAS"}
    df = runner.add_exit_times(df, MarketClock())
    ex = df.groupby("symbol")["exit_time"].first()
    assert ex["CASCO"] == time(15, 0) and ex["NONCAS"] == time(15, 15)

    results = runner.run_study(df, BacktestConfig(slippage_bps=0), ["rules", "random"], strategies.BARRIER_GRID[:2])
    assert len(results) == 4
    rules = next(r for r in results if r["name"] == "rules__atr5_1.0x1.5")
    assert rules["summary"]["trades"] == 20              # 2 stocks x 10 days, one trade each per day
    assert "would_halt_on" in rules["summary"]
    assert rules["trades"]["exit_time"].dt.time.max() <= time(15, 15)
