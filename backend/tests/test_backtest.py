from datetime import time

import numpy as np
import pandas as pd
import pytest

from app.backtest import metrics, strategies
from app.backtest.engine import BacktestConfig, run_backtest
from app.trading.charges import round_trip_charges


def bars_for(symbol, prices, day="2026-11-25", atr=10.0, signal_at=None, side=1, exit_t=time(15, 0)):
    t0 = pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata")
    rows = []
    for k, (o, h, lo, c) in enumerate(prices):
        rows.append({"ts": t0 + pd.Timedelta(minutes=5 * k), "day": pd.Timestamp(day).date(), "symbol": symbol,
                     "open": o, "high": h, "low": lo, "close": c, "atr14": atr, "daily_atr_pct": 0.02,
                     "exit_time": exit_t, "signal": side if k == signal_at else 0, "score": 1.0})
    return pd.DataFrame(rows)


def flat(n, p=100.0):
    return [(p, p + 1, p - 1, p)] * n


CFG = BacktestConfig(capital=100_000, risk_pct=1.0, slippage_bps=0, max_notional_per_trade=5.0)


def test_long_target_trade():
    # signal at candle 3 (closes 09:35) -> entry at candle 4 open = 100; stop 90, target 115
    px = flat(4) + [(100, 101, 99, 100), (100, 120, 99, 118)] + flat(60, 118)
    res = run_backtest(bars_for("AAA", px, signal_at=3), CFG)
    t = res.trades.iloc[0]
    assert (t.side, t.exit_reason, t.entry_price, t.exit_price) == ("long", "target", 100.0, 115.0)
    assert t.qty == 100   # 1% of 1L = 1000 risk / 10 stop
    assert t.charges == pytest.approx(round_trip_charges(100 * 100, 115 * 100).total)
    assert t.net == pytest.approx(1500 - t.charges)
    assert res.daily.iloc[0].equity == pytest.approx(100_000 + t.net)


def test_stop_inside_candle_fills_at_stop_with_slippage():
    cfg = BacktestConfig(capital=100_000, risk_pct=1.0, slippage_bps=10, max_notional_per_trade=5.0)
    px = flat(4) + [(100, 101, 80, 85)] + flat(60, 85)
    t = run_backtest(bars_for("AAA", px, signal_at=3), cfg).trades.iloc[0]
    assert t.exit_reason == "stop"
    entry = 100 * 1.001
    assert t.entry_price == pytest.approx(round(entry, 2))
    assert t.exit_price == pytest.approx(round(round(entry - 10, 2) * 0.999, 2))


def test_both_barriers_in_one_candle_takes_stop():
    px = flat(4) + [(100, 130, 70, 100)] + flat(60)
    t = run_backtest(bars_for("AAA", px, signal_at=3), CFG).trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_price == 90


def test_short_trade_mirrors():
    px = flat(4) + [(100, 101, 80, 82)] + flat(60, 82)
    t = run_backtest(bars_for("AAA", px, signal_at=3, side=-1), CFG).trades.iloc[0]
    assert (t.side, t.exit_reason, t.exit_price) == ("short", "target", 85.0)
    assert t.gross == pytest.approx((100 - 85) * t.qty)


def test_time_exit_at_square_off():
    px = flat(75, 100.0)
    px[68] = (100, 101, 99, 104)          # 14:55 candle, closes at our 15:00 exit
    t = run_backtest(bars_for("AAA", px, signal_at=3), CFG).trades.iloc[0]
    assert t.exit_reason == "time" and t.exit_price == 104
    assert t.exit_time == pd.Timestamp("2026-11-25 15:00", tz="Asia/Kolkata")


def test_one_trade_per_stock_and_daily_cap():
    frames = []
    for i in range(8):
        f = bars_for(f"S{i}", flat(75), signal_at=3)
        f.loc[f.index >= 3, "signal"] = 1          # signal on every candle from 09:35
        frames.append(f)
    res = run_backtest(pd.concat(frames), CFG)
    assert len(res.trades) == 5                     # max_trades_per_day
    assert res.trades["symbol"].is_unique


def test_daily_loss_cap_blocks_new_entries():
    cfg = BacktestConfig(capital=100_000, risk_pct=1.5, slippage_bps=0, daily_loss_cap_pct=2.0,
                         max_notional_per_trade=5.0)
    crash = flat(4) + [(100, 101, 85, 86)] + flat(70, 86)
    a = bars_for("A", crash, signal_at=3)
    b = bars_for("B", crash, signal_at=3)
    c = bars_for("C", flat(75), signal_at=10)        # would come later in the day
    res = run_backtest(pd.concat([a, b, c]), cfg)
    assert set(res.trades["symbol"]) == {"A", "B"}   # 2 stops = -3% > 2% cap -> C blocked


def test_random_strategy_loses_about_its_costs():
    rng = np.random.default_rng(1)
    frames = []
    for s in range(20):
        price, rows = 1000.0, []
        for d in pd.bdate_range("2026-06-01", periods=60):
            t0 = d + pd.Timedelta(hours=9, minutes=15)
            for k in range(75):
                o = price
                c = o * (1 + rng.normal(0, 0.0015))
                rows.append({"ts": (t0 + pd.Timedelta(minutes=5 * k)).tz_localize("Asia/Kolkata"), "day": d.date(),
                             "symbol": f"S{s}", "open": o, "high": max(o, c) * 1.0007, "low": min(o, c) * 0.9993,
                             "close": c, "atr14": c * 0.002, "daily_atr_pct": 0.015, "exit_time": time(15, 0),
                             "exclude": False, "in_universe": True})
                price = c
        frames.append(pd.DataFrame(rows))
    df = strategies.random_baseline(pd.concat(frames, ignore_index=True), rate=0.01)
    res = run_backtest(df, BacktestConfig(slippage_bps=3, max_drawdown_halt_pct=1000))  # measure costs: no halt
    s = metrics.summarize(res.trades, res.daily, 100_000)
    assert s["trades"] > 200
    assert s["net_pnl"] < 0                                   # no edge -> lose
    assert abs(s["gross_pnl"]) < s["charges"] * 3             # loss is mostly costs and slippage
    periods = metrics.by_period(res.trades, res.daily, 100_000)
    assert periods and metrics.consistency(periods) <= 1


def test_rules_baseline_needs_all_conditions():
    df = pd.DataFrame({"rvol": [2.0, 2.0, 1.0], "adx14": [25, 25, 25], "or_breakout": [1, -1, 1],
                       "vwap_dist": [0.01, -0.01, 0.01], "ema50_15m_dist": [0.01, -0.01, 0.01],
                       "st_dir": [1, -1, 1], "nifty_ret_open": [0.0, 0.0, 0.0],
                       "exclude": [False, False, False], "in_universe": [True, True, True]})
    out = strategies.rules_baseline(df)
    assert out["signal"].tolist() == [1, -1, 0]       # third row fails relative volume


def test_kill_switch_stops_trading_after_drawdown():
    days = pd.bdate_range("2026-11-02", periods=8)
    frames = []
    for d in days:   # every day: two stops of 1.5% each = -3% a day
        for sym in ("A", "B"):
            frames.append(bars_for(sym, flat(4) + [(100, 101, 85, 86)] + flat(70, 86), day=str(d.date()),
                                   signal_at=3))
    cfg = BacktestConfig(capital=100_000, risk_pct=1.5, slippage_bps=0, max_notional_per_trade=5.0,
                         max_drawdown_halt_pct=10.0)
    res = run_backtest(pd.concat(frames), cfg)
    traded_days = res.trades["day"].nunique()
    assert traded_days == 4                     # -3%, -6%, -9%, -12% -> halted from day 5
    assert res.halted_on == days[4].date()
    assert res.daily["equity"].min() > 85_000   # never trades into oblivion


def test_slippage_is_accounted():
    cfg = BacktestConfig(capital=100_000, risk_pct=1.0, slippage_bps=10, max_notional_per_trade=5.0)
    px = flat(4) + [(100, 101, 80, 85)] + flat(60, 85)
    res = run_backtest(bars_for("AAA", px, signal_at=3), cfg)
    t = res.trades.iloc[0]
    entry_slip = (round(100 * 1.001, 2) - 100) * t.qty
    stop_ref = round(t.entry_price - 10, 2)
    exit_slip = (stop_ref - t.exit_price) * t.qty
    assert t.slippage == pytest.approx(entry_slip + exit_slip)
    s = metrics.summarize(res.trades, res.daily, 100_000)
    assert s["gross_before_costs"] == pytest.approx(t.gross + t.slippage)
    assert s["cost_per_trade"] == pytest.approx(t.slippage + t.charges, abs=0.01)


def test_halt_day_reported_without_stopping():
    d = pd.DataFrame({"day": pd.bdate_range("2026-01-01", periods=5).date, "net": [-5000, -6000, -5000, 2000, 1000]})
    d["equity"] = 100_000 + d["net"].cumsum()
    # equity 95k, 89k, 84k (16% below 100k peak) -> trading would stop from the next day
    assert metrics.halt_day(d, 15.0) == str(d["day"].iloc[3])
    assert metrics.halt_day(d, 50.0) is None
