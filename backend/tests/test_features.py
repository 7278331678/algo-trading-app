"""Feature builder and labels: correctness on hand-made data, and the no-look-ahead guarantee."""

from datetime import time

import numpy as np
import pandas as pd
import pytest

from app.features import build, labels

rng = np.random.default_rng(11)


def candles(days=30, minutes=5, bars=75, start="2026-09-01", drift=0.0):
    rows = []
    price = 1000.0
    for d in pd.bdate_range(start, periods=days):
        t0 = d + pd.Timedelta(hours=9, minutes=15)
        for k in range(bars):
            o = price
            c = max(1.0, o * (1 + rng.normal(drift, 0.002)))
            h = max(o, c) * (1 + abs(rng.normal(0, 0.001)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.001)))
            rows.append({"ts": (t0 + pd.Timedelta(minutes=minutes * k)).tz_localize("Asia/Kolkata"),
                         "open": o, "high": h, "low": lo, "close": c, "volume": float(rng.integers(1e3, 1e5))})
            price = c
    return build.frame(rows)


def resample(m5, rule):
    g = m5.set_index("ts").resample(rule, origin="start_day", offset="15min")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return build.frame(out.reset_index())


@pytest.fixture(scope="module")
def data():
    m5 = candles(days=40)
    m15 = resample(m5, "15min")
    daily = build.frame(m5.groupby("day").agg(ts=("ts", "first"), open=("open", "first"), high=("high", "max"),
                                              low=("low", "min"), close=("close", "last"),
                                              volume=("volume", "sum")).reset_index(drop=True))
    daily["ts"] = pd.to_datetime(daily["day"]).dt.tz_localize("Asia/Kolkata")
    return m5, m15, daily


def test_all_stock_features_present(data):
    f = build.stock_features(*data)
    assert set(build.STOCK_FEATURES) <= set(f.columns)
    late = f[f["day"] == f["day"].max()]
    assert late[build.STOCK_FEATURES].iloc[10:].notna().all().all()  # warmed up by day 40


def test_decision_time_is_candle_close(data):
    f = build.stock_features(*data)
    assert (f["decision_time"] - f["ts"]).eq(pd.Timedelta(minutes=5)).all()


def test_prev_session_values_come_from_previous_day(data):
    m5, m15, daily = data
    f = build.stock_features(m5, m15, daily)
    d2 = sorted(f["day"].unique())[5]
    prev = daily[daily["day"] < d2].iloc[-1]
    row = f[f["day"] == d2].iloc[0]
    assert row["prev_close"] == pytest.approx(prev["close"])
    assert row["gap_pct"] == pytest.approx(f[f["day"] == d2]["open"].iloc[0] / prev["close"] - 1)


def test_no_look_ahead_by_truncation(data):
    """Features at time T must be identical whether or not later data exists."""
    m5, m15, daily = data
    full = build.stock_features(m5, m15, daily).set_index("ts")
    for cut in [m5["ts"].iloc[1500], m5["ts"].iloc[2222], m5["ts"].iloc[2899]]:
        known_by = cut + pd.Timedelta(minutes=5)   # decision time of the cut candle
        part = build.stock_features(
            m5[m5["ts"] <= cut],
            m15[m15["ts"] + pd.Timedelta(minutes=15) <= known_by],
            daily[pd.to_datetime(daily["day"]) < pd.Timestamp(cut.date())],
        ).set_index("ts")
        a = full.loc[part.index, build.STOCK_FEATURES]
        b = part[build.STOCK_FEATURES]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------ labels
def day_frame(prices, atr=10.0):
    """One session of 5-min candles from (open, high, low, close) tuples."""
    t0 = pd.Timestamp("2026-11-25 09:15", tz="Asia/Kolkata")
    rows = [{"ts": t0 + pd.Timedelta(minutes=5 * k), "open": o, "high": h, "low": lo, "close": c, "volume": 1.0}
            for k, (o, h, lo, c) in enumerate(prices)]
    df = build.frame(rows)
    df["atr14"] = atr
    return df


EXIT = {pd.Timestamp("2026-11-25").date(): time(15, 0)}


def flat(n, p=100.0):
    return [(p, p + 1, p - 1, p)] * n


def test_long_target_hit_first():
    # decision at 09:30 close (index 2); entry 100 at index 3; target 115, stop 90
    px = flat(3) + [(100, 101, 99, 100), (100, 116, 99, 114)] + flat(40, 114)
    lab = labels.triple_barrier(day_frame(px), EXIT)
    assert lab.loc[2, "y_long"] == 1 and lab.loc[2, "ret_long"] == pytest.approx(0.15)
    assert lab.loc[2, "y_short"] == 0 and lab.loc[2, "ret_short"] < 0


def test_same_candle_hits_both_counts_as_stop():
    px = flat(3) + [(100, 120, 85, 100)] + flat(40)
    lab = labels.triple_barrier(day_frame(px), EXIT)
    assert lab.loc[2, "y_long"] == 0 and lab.loc[2, "ret_long"] == pytest.approx(-0.10)


def test_gap_through_stop_fills_at_open():
    px = flat(3) + [(100, 101, 99, 100), (80, 81, 79, 80)] + flat(40, 80)
    lab = labels.triple_barrier(day_frame(px), EXIT)
    assert lab.loc[2, "ret_long"] == pytest.approx(-0.20)  # filled at 80, worse than the 90 stop


def test_timeout_exits_at_exit_time_close():
    px = flat(75, 100.0)
    px[68] = (100, 101, 99, 103)   # 14:55 candle closes at 15:00 = exit time
    lab = labels.triple_barrier(day_frame(px), EXIT)
    assert lab.loc[2, "y_long"] == 0 and lab.loc[2, "ret_long"] == pytest.approx(0.03)


def test_labels_only_inside_entry_window():
    lab = labels.triple_barrier(day_frame(flat(75)), EXIT)
    t = day_frame(flat(75))["ts"].dt.time
    end = (day_frame(flat(75))["ts"] + pd.Timedelta(minutes=5)).dt.time
    inside = (end >= time(9, 30)) & (end <= time(14, 30))
    assert lab.loc[~inside, "y_long"].isna().all()
    assert lab.loc[inside, "y_long"].notna().all()
    assert t.iloc[0] == time(9, 15)


def test_labels_never_use_candles_after_exit_time():
    px = flat(75)
    px[70] = (100, 200, 99, 100)   # 15:05 candle: after our 15:00 exit, must be ignored
    lab = labels.triple_barrier(day_frame(px), EXIT)
    assert lab["y_long"].max() == 0


def test_builder_ignores_future_15m_and_daily_even_when_given(data):
    """Hand the builder the FULL 15-min and daily history: its own time logic must still
    only use 15-min candles closed by decision time and the previous session's daily candle."""
    m5, m15, daily = data
    full = build.stock_features(m5, m15, daily).set_index("ts")
    cut = m5["ts"].iloc[2222]
    part = build.stock_features(m5[m5["ts"] <= cut], m15, daily).set_index("ts")
    pd.testing.assert_frame_equal(full.loc[part.index, build.STOCK_FEATURES], part[build.STOCK_FEATURES],
                                  check_exact=False, rtol=1e-9, atol=1e-12)


def test_exclusions():
    from datetime import date, timedelta

    from app.features import exclusions

    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]
    r = exclusions.excluded_days(days, {days[40]}, {days[50]}, after=3, warmup=5)
    assert all(r[d] == "warmup" for d in days[:5])
    assert all(r[d] == "corporate_action" for d in days[40:44]) and days[44] not in r
    assert r[days[50]] == "special_session"
    assert exclusions.load_corporate_actions()["VEDL"] == {date(2026, 4, 30)}
