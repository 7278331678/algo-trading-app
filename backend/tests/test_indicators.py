"""Indicators checked against independent, slow, textbook loop implementations."""

import math

import numpy as np
import pandas as pd
import pytest

from app.features import indicators as ind

rng = np.random.default_rng(7)


def session_ts(days=4, bars=75, freq_min=5, start="2026-11-02"):
    out = []
    for d in pd.bdate_range(start, periods=days):
        out += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=bars,
                                  freq=f"{freq_min}min", tz="Asia/Kolkata"))
    return pd.Series(out)


def ohlcv(n=300):
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    open_ = close + rng.normal(0, 0.5, n)
    vol = rng.integers(1_000, 50_000, n).astype(float)
    ts = session_ts(days=math.ceil(n / 75)).iloc[:n].reset_index(drop=True)
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol})


def loop_ema(x, n):
    a, out, prev = 2 / (n + 1), [], None
    for v in x:
        prev = v if prev is None else a * v + (1 - a) * prev
        out.append(prev)
    return np.array(out)


def loop_wilder(x, n):
    out, prev = [], None
    for v in x:
        prev = v if prev is None else prev + (v - prev) / n
        out.append(prev)
    return np.array(out)


def test_ema_matches_recursive_definition():
    df = ohlcv()
    got = ind.ema(df.close, 9).to_numpy()
    want = loop_ema(df.close.to_numpy(), 9)
    assert np.isnan(got[:8]).all()
    np.testing.assert_allclose(got[8:], want[8:], rtol=1e-12)


def test_rsi_matches_wilder_loop():
    df = ohlcv()
    c = df.close.to_numpy()
    d = np.diff(c)
    g = loop_wilder(np.clip(d, 0, None), 14)
    lo = loop_wilder(np.clip(-d, 0, None), 14)
    want = 100 - 100 / (1 + g / lo)
    got = ind.rsi(df.close, 14).to_numpy()[1:]
    np.testing.assert_allclose(got[13:], want[13:], rtol=1e-10)
    assert ((got[13:] >= 0) & (got[13:] <= 100)).all()


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(s, 14).dropna().eq(100).all()


def test_atr_matches_loop():
    df = ohlcv()
    h, lo, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    tr = [h[0] - lo[0]] + [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))]
    want = loop_wilder(np.array(tr), 14)
    got = ind.atr(df.high, df.low, df.close, 14).to_numpy()
    np.testing.assert_allclose(got[13:], want[13:], rtol=1e-10)


def test_macd_relationships():
    df = ohlcv()
    m = ind.macd(df.close)
    line = loop_ema(df.close.to_numpy(), 12) - loop_ema(df.close.to_numpy(), 26)
    np.testing.assert_allclose(m.macd.to_numpy()[25:], line[25:], rtol=1e-10)
    np.testing.assert_allclose(m.macd_hist, m.macd - m.macd_signal, equal_nan=True)


def test_adx_bounds_and_trend():
    n = 200
    up = pd.Series(np.linspace(100, 200, n))
    a = ind.adx(up + 0.5, up - 0.5, up, 14).dropna()
    assert ((a.adx >= 0) & (a.adx <= 100)).all()
    assert a.adx.iloc[-1] > 50 and (a.plus_di > a.minus_di).iloc[-1]  # strong steady uptrend


def test_supertrend_follows_trend_and_flips():
    up = np.linspace(100, 150, 80)
    down = np.linspace(150, 90, 80)
    c = pd.Series(np.concatenate([up, down]))
    st = ind.supertrend(c + 0.5, c - 0.5, c, 10, 3.0)
    assert st.st_dir.iloc[60] == 1 and st.st_line.iloc[60] < c.iloc[60]
    assert st.st_dir.iloc[-1] == -1 and st.st_line.iloc[-1] > c.iloc[-1]


def test_bollinger_width_matches_definition():
    df = ohlcv()
    w = ind.bollinger_width(df.close, 20, 2.0)
    win = df.close.iloc[100:120]
    assert w.iloc[119] == pytest.approx(4 * win.std(ddof=0) / win.mean())


def test_vwap_resets_each_session():
    df = ohlcv(150)  # spans two sessions (75 five-minute bars each)
    day = ind.session_key(df.ts)
    v = ind.vwap(df.high, df.low, df.close, df.volume, day)
    tp = (df.high + df.low + df.close) / 3
    first_of_day2 = day.ne(day.shift()).to_numpy().nonzero()[0][1]
    assert v.iloc[0] == pytest.approx(tp.iloc[0])
    assert v.iloc[first_of_day2] == pytest.approx(tp.iloc[first_of_day2])
    sl = slice(first_of_day2, first_of_day2 + 10)
    assert v.iloc[first_of_day2 + 9] == pytest.approx((tp[sl] * df.volume[sl]).sum() / df.volume[sl].sum())


def test_obv():
    c = pd.Series([10, 11, 10, 10, 12.0])
    v = pd.Series([100, 200, 300, 400, 500.0])
    assert ind.obv(c, v).tolist() == [0, 200, -100, -100, 400]


def test_relative_volume_uses_only_past_sessions():
    days = pd.date_range("2026-10-01 09:15", periods=30, freq="D", tz="Asia/Kolkata")
    ts = pd.Series(days)
    vol = pd.Series([100.0] * 29 + [300.0])
    rv = ind.relative_volume(vol, ts, sessions=20)
    assert rv.iloc[-1] == pytest.approx(3.0)
    assert math.isnan(rv.iloc[3])  # not enough history


def test_opening_range_unknown_until_window_closes():
    df = ohlcv(75)
    day = ind.session_key(df.ts)
    o = ind.opening_range(df.high, df.low, df.ts, day, minutes=15)
    assert o.or_high.iloc[:3].isna().all()  # 09:15, 09:20, 09:25
    assert o.or_high.iloc[3] == pytest.approx(df.high.iloc[:3].max())
    assert o.or_low.iloc[50] == pytest.approx(df.low.iloc[:3].min())


def test_cpr_ordering():
    c = ind.cpr(pd.Series([110.0]), pd.Series([100.0]), pd.Series([101.0]))
    assert c.cpr_bottom.iloc[0] <= c.cpr_pivot.iloc[0] <= c.cpr_top.iloc[0]
