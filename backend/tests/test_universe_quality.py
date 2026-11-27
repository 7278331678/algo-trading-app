from datetime import date, datetime, timedelta

import pytest

from app.core.market_clock import IST
from app.marketdata import quality, universe

CSV = (
    "﻿Company Name,Industry,Symbol,Series,ISIN Code\n"
    "Infosys Ltd.,Information Technology,INFY,EQ,INE009A01021\n"
    "HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034\n"
    "Some Co Ltd.,Diversified,SOMECO,EQ,INE000000000\n"
)


def test_parse_constituents():
    rows = universe.parse_constituents_csv(CSV)
    assert [r.symbol for r in rows] == ["INFY", "HDFCBANK", "SOMECO"]
    assert rows[0].sector_index == "NIFTY IT"
    assert rows[1].sector_index == "NIFTY FIN SERVICE"
    assert rows[2].sector_index == universe.BROAD_INDEX  # unknown industry -> NIFTY 50


def test_parse_rejects_wrong_file():
    with pytest.raises(ValueError):
        universe.parse_constituents_csv("<html>blocked</html>\n")


def test_parse_rejects_duplicates():
    with pytest.raises(ValueError):
        universe.parse_constituents_csv(CSV + "Infosys Ltd.,Information Technology,INFY,EQ,X\n")


def test_diff_membership():
    added, removed, kept = universe.diff_membership({"A", "B", "C"}, {"B", "C", "D"})
    assert added == {"D"} and removed == {"A"} and kept == {"B", "C"}


# ------------------------------------------------------------------ quality
def candle(ts, o=100, h=101, lo=99, c=100.5, v=1000):
    return {"ts": ts, "open": o, "high": h, "low": lo, "close": c, "volume": v}


def session(day, interval_min=15, n=None, **kw):
    start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
    count = n if n is not None else 375 // interval_min
    return [candle(start + timedelta(minutes=interval_min * k), **kw) for k in range(count)]


DAYS = {date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25)}


def test_clean_series_has_no_issues():
    rows = [c for d in sorted(DAYS) for c in session(d)]
    assert quality.run_checks(rows, "15minute", DAYS) == []


def test_missing_day_is_critical():
    rows = [c for d in sorted(DAYS - {date(2026, 11, 24)}) for c in session(d)]
    issues = quality.run_checks(rows, "15minute", DAYS)
    assert any(i.issue == "missing_day" and i.severity == "critical" for i in issues)


def test_incomplete_day_levels():
    d = date(2026, 11, 25)
    crit = quality.check_completeness(session(PREV) + session(d, n=20), "15minute", {PREV, d})
    assert crit[0].severity == "critical"
    warn = quality.check_completeness(session(PREV, interval_min=1) + session(d, interval_min=1, n=365),
                                      "minute", {PREV, d})
    assert warn[0].severity == "warning"


def test_bad_ohlc_and_duplicates():
    ts = datetime(2026, 11, 25, 9, 15, tzinfo=IST)
    rows = [candle(ts, h=98), candle(ts + timedelta(minutes=15), o=0)]
    kinds = {i.issue for i in quality.check_ohlc(rows)}
    assert kinds == {"ohlc_inconsistent", "non_positive_price"}
    assert quality.check_duplicates([candle(ts), candle(ts)])[0].issue == "duplicate_candle"


def test_zero_volume_only_for_stocks():
    d = date(2026, 11, 25)
    rows = session(PREV) + session(d, v=0)
    assert any(i.issue == "zero_volume" for i in quality.check_completeness(rows, "15minute", {PREV, d}))
    assert quality.check_completeness(rows, "15minute", {PREV, d}, is_index=True) == []


def test_price_jump_flags_split():
    d1 = datetime(2026, 11, 24, tzinfo=IST)
    d2 = datetime(2026, 11, 25, tzinfo=IST)
    rows = [candle(d1, o=1000, h=1010, lo=990, c=1000), candle(d2, o=500, h=505, lo=495, c=500)]
    issues = quality.check_jumps(rows)
    assert issues and issues[0].severity == "critical"
    assert quality.run_checks(rows, "day", {d1.date(), d2.date()}, is_index=True) == []


def test_no_data_at_all():
    assert quality.check_completeness([], "day", DAYS)[0].issue == "no_data"


def test_migration_splits_into_single_statements():
    from app.db import MIGRATIONS_DIR, split_sql

    stmts = split_sql((MIGRATIONS_DIR / "001_data_layer.sql").read_text())
    assert len(stmts) == 12  # 6 tables + 6 indexes
    assert all(";" not in s and not s.startswith("--") for s in stmts)


def _times(day_n, minutes):
    return [(datetime(2026, 1, 1, 9, 15) + timedelta(minutes=minutes * k)).time() for k in range(day_n)]


PREV = date(2026, 11, 7)  # an earlier day so the checked day is not the series' first day


def test_special_session_skipped():
    d = date(2026, 11, 8)
    rows = session(PREV) + session(d, n=2)
    ref = {PREV: _times(25, 15), d: _times(4, 15)}
    assert quality.check_completeness(rows, "15minute", {PREV, d}, reference_times=ref) == []


def test_market_wide_short_day_is_not_flagged_but_stock_gap_is():
    d = date(2026, 11, 25)
    ref = {PREV: _times(75, 5), d: _times(72, 5)}  # the index itself had 72/75 that day
    ok = session(PREV, interval_min=5) + session(d, interval_min=5, n=72)
    assert quality.check_completeness(ok, "5minute", {PREV, d}, reference_times=ref) == []
    gap = session(PREV, interval_min=5) + session(d, interval_min=5, n=69)
    assert quality.check_completeness(gap, "5minute", {PREV, d}, reference_times=ref)[0].issue == "incomplete_day"


def test_auction_stock_ending_1515_is_complete_but_non_auction_must_run_to_1530():
    from datetime import time as t
    d = date(2026, 11, 25)
    ref = {PREV: _times(75, 5), d: _times(75, 5)}
    auction = {PREV: t(15, 15), d: t(15, 15)}
    cas_stock = session(PREV, interval_min=5, n=72) + session(d, interval_min=5, n=72)   # stops 15:10
    assert quality.check_completeness(cas_stock, "5minute", {PREV, d}, reference_times=ref,
                                      auction_end=auction) == []
    # a stock trading past 15:15 that day is held to the full session
    late = session(PREV, interval_min=5) + session(d, interval_min=5, n=74)
    assert quality.check_completeness(late, "5minute", {PREV, d}, reference_times=ref, auction_end=auction) == []
    short = session(PREV, interval_min=5) + [c for c in session(d, interval_min=5) if c["ts"].minute != 30][:70]
    issues = quality.check_completeness(short, "5minute", {PREV, d}, reference_times=ref, auction_end=auction)
    assert issues and issues[0].issue == "incomplete_day"


def test_listing_day_not_checked():
    d1, d2 = date(2026, 11, 24), date(2026, 11, 25)
    rows = session(d1, n=15) + session(d2)   # first day opened late after the listing auction
    assert quality.check_completeness(rows, "15minute", {d1, d2}) == []


def test_history_seed_is_consistent():
    rows = universe.load_history_seed()
    assert len(rows) >= 50
    by_sym = {}
    for r in rows:
        assert r["valid_to"] is None or r["valid_to"] >= r["valid_from"]
        by_sym.setdefault(r["symbol"], []).append(r)
    for sym, spells in by_sym.items():   # no overlapping spells for one stock
        spells.sort(key=lambda r: r["valid_from"])
        for a, b in zip(spells, spells[1:], strict=False):
            assert a["valid_to"] is not None and a["valid_to"] < b["valid_from"], sym
    # round trips we know about
    assert [r["valid_to"] is None for r in sorted(by_sym["UNIONBANK"], key=lambda r: r["valid_from"])] == [False, True]
    assert [r["valid_to"] is None for r in sorted(by_sym["MUTHOOTFIN"], key=lambda r: r["valid_from"])] == [False, True]
