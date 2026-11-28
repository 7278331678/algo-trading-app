"""Phase 5 step 1: NSE event parsing, categories, incremental fetching and point-in-time features."""

import urllib.error
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.core.market_clock import IST
from app.events import classify, features, nse, store


# ------------------------------------------------------------------ categories
@pytest.mark.parametrize("subject,body,cat", [
    ("News Verification", "The Exchange has sought clarification", "clarification"),
    ("Action(s) taken or orders passed", None, "regulatory"),
    ("Other Restructuring", None, "mna"),
    ("Trading Window", "SEBI (Prohibition of Insider Trading) Regulations", "routine"),   # boilerplate text
    ("Analysts/Institutional Investor Meet/Con. Call Updates", "Transcript of the financial results call", "routine"),
    ("Financial Result Updates", None, "results"),
    ("Outcome of Board Meeting", "Approved the unaudited financial results for the quarter", "results"),
    ("Award of Order/Receipt of Order", None, "order"),
    ("Credit Rating", None, "rating"),
    ("Acquisition", None, "mna"),
    ("Clarification - Financial Results", None, "clarification"),     # exchange query beats 'results'
    ("Trading Window-XBRL", None, "routine"),
    ("Updates", "Receipt of order worth Rs 450 crore from NHAI", "order"),  # routine subject, real news in body
    ("Analysts/Institutional Investor Meet/Con. Call Updates", None, "routine"),
    ("Something new", None, "other"),
])
def test_classify(subject, body, cat):
    assert classify.classify(subject, body) == cat


def test_results_meeting_detection():
    assert classify.is_results_meeting("Financial Results")
    assert not classify.is_results_meeting("Fund Raising")


# ------------------------------------------------------------------ parsing NSE payloads
ANN = [
    {"symbol": "LAXMIMACH", "desc": "Press Release", "dt": "19072025101722", "an_dt": "19-Jul-2025 10:18:09",
     "sort_date": "2025-07-19 10:18:09", "seq_id": "1487462", "attchmntText": "Business update",
     "attchmntFile": "https://nsearchives.nseindia.com/corporate/x.pdf"},
    {"symbol": "LAXMIMACH", "desc": "Credit Rating", "an_dt": "18-Jul-2025 19:02:00"},
    {"symbol": "LAXMIMACH", "desc": "No time"},
]


def test_parse_announcements():
    rows, skipped = nse.parse_announcements({"data": ANN}, "LAXMIMACH")
    assert skipped == 1 and len(rows) == 2
    a, b = rows
    assert a["public_at"] == datetime(2025, 7, 19, 10, 18, 9, tzinfo=IST)
    assert a["ext_id"] == "1487462" and a["category"] == "guidance"
    assert b["category"] == "rating" and len(b["ext_id"]) == 20     # hashed id when NSE gives none
    assert nse.parse_announcements(ANN, "X")[0][0]["symbol"] == "LAXMIMACH"   # bare list also works


def test_parse_board_meetings_estimates_known_time_late():
    payload = [{"bm_symbol": "TCS", "bm_date": "09-Oct-2025", "bm_purpose": "Financial Results",
                "bm_timestamp": "25-Sep-2025 17:10:00"},
               {"bm_symbol": "TCS", "bm_date": "08-Jan-2026", "bm_purpose": "Financial Results/Dividend"},
               {"bm_symbol": "TCS", "bm_purpose": "no date"}]
    rows, skipped = nse.parse_board_meetings(payload, "TCS")
    assert skipped == 1
    assert rows[0]["known_at"] == datetime(2025, 9, 25, 17, 10, tzinfo=IST) and not rows[0]["known_at_estimated"]
    assert rows[1]["known_at"] == datetime(2026, 1, 6, tzinfo=IST) and rows[1]["known_at_estimated"]
    assert all(r["is_results"] for r in rows)


def test_windows_and_urls():
    w = nse.windows(date(2025, 1, 1), date(2025, 7, 1), 90)
    assert w[0] == (date(2025, 1, 1), date(2025, 3, 31)) and w[-1][1] == date(2025, 7, 1)
    assert all(b + timedelta(days=1) == a for (_, b), (a, _) in zip(w, w[1:], strict=False))
    url = nse.announcements_url("M&M", date(2025, 1, 1), date(2025, 3, 31))
    assert "symbol=M%26M" in url and "from_date=01-01-2025" in url and "to_date=31-03-2025" in url


def test_plan_skips_done_windows_but_refreshes_recent_days():
    end = date(2026, 11, 27)
    all_w = nse.windows(date(2026, 1, 1), end, 90)
    done = {("TCS", a, b) for a, b in all_w}
    todo = store.plan(["TCS", "INFY"], date(2026, 1, 1), end, done, 90)
    assert [t for t in todo if t[0] == "TCS"] == [("TCS", *all_w[-1])]      # only the window touching today
    assert len([t for t in todo if t[0] == "INFY"]) == len(all_w)


# ------------------------------------------------------------------ polite session behaviour
class _Resp(Exception):
    pass


def _session(responses):
    s = nse.NseSession(rate_per_sec=1000)
    calls = []

    def fake_open(url):
        calls.append(url)
        r = responses.pop(0) if not url.endswith(("/", "corporate-filings-announcements")) else b"<html>"
        if isinstance(r, int):
            raise urllib.error.HTTPError(url, r, "x", {}, None)
        return r

    s._open = fake_open
    return s, calls


def test_session_refreshes_cookies_once_then_succeeds():
    s, calls = _session([403, b'{"data": []}'])
    assert s.get_json("https://www.nseindia.com/api/x") == {"data": []}
    assert sum(c.endswith("/") for c in calls) == 2       # warmed up twice (initial + after 403)


def test_session_stops_when_blocked():
    s, _ = _session([403, 403])
    with pytest.raises(nse.NseBlocked):
        s.get_json("https://www.nseindia.com/api/x")
    s, _ = _session([b"<html>captcha</html>"])
    with pytest.raises(nse.NseBlocked):
        s.get_json("https://www.nseindia.com/api/x")


# ------------------------------------------------------------------ point-in-time features
def _rows():
    days = [date(2026, 11, 24), date(2026, 11, 25)]
    ts = []
    for d in days:
        ts += list(pd.date_range(pd.Timestamp(d).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15),
                                 periods=75, freq="5min"))
    df = pd.DataFrame({"ts": ts, "symbol": "TCS"})
    df["decision_time"] = df["ts"] + pd.Timedelta(minutes=5)
    df["day"] = df["ts"].dt.date
    return df


def _at(df, day, hhmm):
    t = pd.Timestamp(f"{day} {hhmm}").tz_localize(IST)
    return df.index[df["decision_time"] == t][0]


def _ann(*items):
    return pd.DataFrame([{"symbol": "TCS", "public_at": pd.Timestamp(t).tz_localize(IST), "category": c}
                         for t, c in items])


NO_BM = pd.DataFrame(columns=["symbol", "meeting_date", "is_results", "known_at"])


def test_overnight_news_counts_and_latency():
    ann = _ann(("2026-11-24 15:00", "order"),      # before yesterday's close: not "since close"
               ("2026-11-24 18:30", "results"),    # after the close: overnight news
               ("2026-11-25 09:40", "rating"),
               ("2026-11-25 09:44:30", "mna"),     # 30s before a 09:45 decision: too fresh to act on
               ("2026-11-25 08:00", "routine"))
    df = features.add_features(_rows(), ann, NO_BM)
    r = df.loc[_at(df, "2026-11-25", "09:45")]
    assert r["ev_n_since_close"] == 3                 # results, routine, rating
    assert r["ev_material_since_close"] == 2
    assert r["ev_results_since_close"] == 1 and r["ev_rating_since_close"] == 1
    assert r["ev_mna_since_close"] == 0 and r["ev_order_since_close"] == 0
    assert r["ev_material_7d"] == 3                   # order, results, rating
    assert r["ev_hours_since_material"] == pytest.approx(4 / 60, abs=1e-3)   # rating 09:40 -> 09:44 cutoff
    later = df.loc[_at(df, "2026-11-25", "09:50")]
    assert later["ev_mna_since_close"] == 1
    first = df.loc[_at(df, "2026-11-24", "09:45")]
    assert first["ev_n_since_close"] == 0 and first["ev_hours_since_material"] == features.HOURS_CAP


def test_results_calendar_respects_when_it_became_known():
    bm = pd.DataFrame([
        {"symbol": "TCS", "meeting_date": date(2026, 11, 25), "is_results": True,
         "known_at": pd.Timestamp("2026-11-24 12:00").tz_localize(IST)},
        {"symbol": "TCS", "meeting_date": date(2026, 11, 27), "is_results": True,
         "known_at": pd.Timestamp("2026-11-25 11:00").tz_localize(IST)},   # announced mid-morning on the 25th
    ])
    df = features.add_features(_rows(), _ann(), bm)
    early = df.loc[_at(df, "2026-11-24", "09:45")]
    assert early["ev_days_to_results"] == features.DAYS_TO_CAP      # nothing announced yet at 09:45
    noon = df.loc[_at(df, "2026-11-24", "12:05")]
    assert noon["ev_days_to_results"] == 1
    on_day = df.loc[_at(df, "2026-11-25", "09:45")]
    assert on_day["ev_results_today"] == 1 and on_day["ev_days_to_results"] == 0
    assert on_day["ev_days_since_results"] == features.DAYS_SINCE_CAP


def test_future_events_never_change_past_features():
    rng = np.random.default_rng(0)
    base = _ann(("2026-11-24 18:30", "results"), ("2026-11-25 09:20", "order"))
    cut = pd.Timestamp("2026-11-25 11:00").tz_localize(IST)
    future = pd.DataFrame({"symbol": "TCS", "category": rng.choice(classify.CATEGORIES, 50),
                           "public_at": [cut + pd.Timedelta(minutes=int(m)) for m in rng.integers(1, 300, 50)]})
    a = features.add_features(_rows(), base, NO_BM)
    b = features.add_features(_rows(), pd.concat([base, future], ignore_index=True), NO_BM)
    past = a["decision_time"] <= cut
    pd.testing.assert_frame_equal(a.loc[past, features.EVENT_FEATURES], b.loc[past, features.EVENT_FEATURES])
    assert not a.loc[~past, features.EVENT_FEATURES].equals(b.loc[~past, features.EVENT_FEATURES])


def test_no_events_at_all_gives_neutral_values():
    df = features.add_features(_rows(), _ann(), NO_BM)
    assert (df["ev_n_since_close"] == 0).all() and (df["ev_hours_since_material"] == features.HOURS_CAP).all()
    assert set(features.EVENT_FEATURES) <= set(df.columns)
    assert all(df[c].dtype == np.float32 for c in features.EVENT_FEATURES)


# ------------------------------------------------------------------ fetch loop
class _FakeConn:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, q, params=None):
        sql = str(q)
        if sql.startswith("SELECT symbol, from_day"):
            return iter(self.db["log"])
        if "corporate_announcements" in sql:
            self.db["ann"] += params
        elif "event_fetch_log" in sql:
            self.db["logged"].append(params)
        return iter(())


class _FakeEngine:
    def __init__(self):
        self.db = {"log": [], "ann": [], "logged": []}

    def connect(self):
        return _FakeConn(self.db)

    begin = connect


def test_fetch_loop_survives_one_bad_window_and_stops_on_block(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "sqlalchemy", types.SimpleNamespace(text=lambda q: q))
    eng = _FakeEngine()

    def get_json(url):
        if "symbol=BAD" in url:
            raise TimeoutError("slow")
        return {"data": ANN[:1]}

    stats = store.fetch_events(eng, get_json, ["TCS", "BAD"], date(2026, 10, 1), date(2026, 11, 27),
                               kinds=("announcements",))
    assert stats["announcements"] == 1 and len(stats["failures"]) == 1 and len(eng.db["logged"]) == 1

    def blocked(url):
        raise nse.NseBlocked("403")

    with pytest.raises(nse.NseBlocked):
        store.fetch_events(_FakeEngine(), blocked, ["TCS"], date(2026, 10, 1), date(2026, 11, 27),
                           kinds=("announcements",))


def test_event_features_flow_through_the_study():
    """Synthetic market where an overnight 'order' filing precedes up-trend days: the models must use it."""
    from datetime import time

    from app.backtest.engine import BacktestConfig
    from app.core.market_clock import MarketClock
    from app.features import build
    from app.models import dataset, study

    from .test_models import synthetic_market

    df = synthetic_market(n_sym=4, days=330, seed=13)
    df["or_breakout"] = np.random.default_rng(2).normal(0, 1, len(df)).astype("float32")   # hide the old hint
    # the hidden daily trend is visible in the last candle vs the first: use it to plant news on up days
    day_move = df.groupby(["symbol", "day"])["close"].agg(["first", "last"])
    up = day_move[day_move["last"] > day_move["first"]].index
    rng = np.random.default_rng(4)
    up_set = set(up)
    picks = [k for k in day_move.index if rng.random() < (0.6 if k in up_set else 0.15)]  # news is a hint, not a law
    ann = pd.DataFrame([{"symbol": s, "category": "order",
                         "public_at": pd.Timestamp(d).tz_localize(IST) - pd.Timedelta(hours=3)} for s, d in picks])
    features.add_features(df, ann, NO_BM)
    res = study.run_study(df, MarketClock(), BacktestConfig(capital=100_000), "daily_0.6x1.2",
                          dataset.CostModel(slippage_bps_per_side=2.0), ("logit",), decision_times=[time(9, 45)],
                          features=build.ALL_FEATURES + features.EVENT_FEATURES)
    ens = [m["auc"] for m in res["fold_metrics"] if m["model"] == "ensemble" and m["side"] == "long"]
    assert np.mean(ens) > 0.6
    assert res["event_rows"]["rows"] > 0 and res["event_rows"]["long/ensemble"] is not None
    assert "ev_order_since_close" in res["_features"]


def test_migrations_split_into_complete_statements():
    """Regression: a ';' inside an inline comment once cut a CREATE TABLE in half."""
    from app.db import MIGRATIONS_DIR, split_sql

    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        for stmt in split_sql(path.read_text(encoding="utf-8")):
            code = "\n".join(line.split("--", 1)[0] for line in stmt.splitlines())
            assert code.count("(") == code.count(")"), f"{path.name}: unbalanced statement: {stmt[:80]}"


# ------------------------------------------------------------------ NSE official RSS
FEED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>NSE News - Latest Announcements</title>
<item><title>Power Mech Projects Limited</title>
<link>https://nsearchives.nseindia.com/corporate/POWERMECH_28112026082052_CL.pdf</link>
<description>Power Mech Projects Limited has informed the Exchange about Copy of Newspaper Publication related to notice
of Postal Ballot |SUBJECT: Copy of Newspaper Publication</description>
<pubDate>28-Nov-2026 08:21:47</pubDate></item>
<item><title>KEC International Limited</title>
<link>https://nsearchives.nseindia.com/corporate/KEC_28112026071011_PressRelease.pdf</link>
<description>KEC International wins New Orders of Rs. 1,407 crores
|SUBJECT: Bagging/Receiving of orders/contracts</description>
<pubDate>28-Nov-2026 07:11:16</pubDate></item>
<item><title>Mahindra &amp; Mahindra Limited</title>
<link>https://nsearchives.nseindia.com/corporate/M%26M_27112026190000_Update.pdf</link>
<description>Sales update |SUBJECT: Updates</description><pubDate>27-Nov-2026 19:00:00</pubDate></item>
<item><title>No Link Company Limited</title><link></link>
<description>Something |SUBJECT: Credit Rating</description><pubDate>27-Nov-2026 18:00:00</pubDate></item>
<item><title>Broken</title><pubDate>yesterday</pubDate></item>
</channel></rss>"""


def test_rss_parse_real_feed_shape():
    from app.events import rss

    by_name = rss.name_index({"No Link Company Ltd.": "NOLINK"})
    rows, bad = rss.parse_announcements(FEED, by_name)
    assert bad == 1 and [r["symbol"] for r in rows] == ["POWERMECH", "KEC", "M&M", "NOLINK"]
    pm, kec, mm, nl = rows
    assert pm["category"] == "routine" and kec["category"] == "order"
    assert kec["public_at"] == datetime(2026, 11, 28, 7, 11, 16, tzinfo=IST)
    assert kec["subject"] == "Bagging/Receiving of orders/contracts"
    assert kec["body"].startswith("KEC International wins")
    assert kec["ext_id"] == "KEC_28112026071011_PressRelease.pdf"     # the attachment name: same id from any source
    assert mm["ext_id"] == "M&M_27112026190000_Update.pdf"
    assert nl["category"] == "rating" and len(nl["ext_id"]) == 20


def test_rss_poll_flags_possible_gaps(monkeypatch):
    import sys
    import types

    from app.events import rss

    monkeypatch.setitem(sys.modules, "sqlalchemy", types.SimpleNamespace(text=lambda q: q))

    class Conn:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, q, params=None):
            if q.startswith("SELECT max(newest)"):
                return types.SimpleNamespace(scalar=lambda: self.db["newest"])
            if "rss_polls" in q:
                self.db["polls"].append(params)
                self.db["newest"] = params["n"]
            elif "corporate_announcements" in q:
                self.db["saved"] += params
            return iter(())

    class Engine:
        def __init__(self):
            self.db = {"newest": None, "polls": [], "saved": []}

        def connect(self):
            return Conn(self.db)

        begin = connect

    eng = Engine()
    st = rss.poll_once(eng, lambda url: FEED, {"KEC", "M&M"}, {})
    assert st == {"items": 3, "kept": 2, "unparsed": 2, "gap": False}     # no name map: NOLINK unparsed
    assert {r["symbol"] for r in eng.db["saved"]} == {"KEC", "M&M"}
    assert not rss.poll_once(eng, lambda url: FEED, {"KEC"}, {})["gap"]   # overlapping polls: no gap
    eng.db["newest"] = datetime(2026, 11, 27, 9, 0, tzinfo=IST)           # last poll ended long before this feed
    assert rss.poll_once(eng, lambda url: FEED, {"KEC"}, {})["gap"]


def test_poll_interval_is_tight_in_market_hours():
    from app.events.rss import poll_interval

    assert poll_interval(datetime(2026, 11, 30, 9, 0, tzinfo=IST)) == 60
    assert poll_interval(datetime(2026, 11, 30, 23, 0, tzinfo=IST)) == 300


# ------------------------------------------------------------------ NSE CSV export (history)
SAMPLE = __import__("pathlib").Path(__file__).parent / "data" / "CF-AN-equities-RELIANCE-sample.csv"


def test_import_real_nse_export():
    from app.events import importer

    rows, bad = importer.parse_csv(SAMPLE.read_bytes())
    assert bad == 0 and len(rows) == 165 and {r["symbol"] for r in rows} == {"RELIANCE"}
    first = rows[0]
    assert first["public_at"] == datetime(2026, 11, 19, 18, 35, 26, tzinfo=IST)     # DISSEMINATION, not receipt
    assert first["ext_id"] == "kavinavora_19112026183441_SE_19112026.pdf"
    no_attachment = [r for r in rows if r["attachment"] is None]
    assert no_attachment and all(len(r["ext_id"]) == 20 for r in no_attachment)
    cats = {r["category"] for r in rows}
    assert {"results", "rating", "mna", "clarification", "regulatory", "guidance", "routine"} <= cats
    results = sorted(r["public_at"].date() for r in rows if r["category"] == "results")
    assert date(2026, 10, 30) in results and date(2026, 1, 16) in results


def test_results_calendar_from_board_meeting_notices():
    from app.events import importer

    rows, _ = importer.parse_csv(SAMPLE.read_bytes())
    m = {(r["meeting_date"], r["known_at"].date()) for r in importer.meeting_rows(rows)}
    assert {(date(2026, 1, 16), date(2026, 1, 8)), (date(2026, 4, 24), date(2026, 4, 17)),
            (date(2026, 7, 17), date(2026, 7, 10)), (date(2026, 10, 30), date(2026, 10, 27))} <= m


@pytest.mark.parametrize("text,expected", [
    ("Board Meeting Intimation. XYZ has informed the Exchange about Board Meeting to be held on 06-Feb-2026 "
     "to inter-alia consider and approve the Unaudited Financial Results", date(2026, 2, 6)),
    ("The meeting of the Board of Directors is scheduled on 5th February, 2026 to approve the results",
     date(2026, 2, 5)),
    ("Board of Directors will be held on 05/02/2026 to consider financial results", date(2026, 2, 5)),
    ("Board meeting scheduled on February 5, 2026 to consider fund raising", None),         # not results
    ("Outcome of Board Meeting: results approved at the meeting held on February 5, 2026", None),
    ("The Board meeting scheduled on January 2, 2025 approved results", None),              # in the past
])
def test_results_meeting_date_formats(text, expected):
    from app.events import importer

    subject, _, body = text.partition(": ")
    if not body:
        subject, body = "", text
    assert importer.results_meeting(subject, body, datetime(2026, 1, 20, 18, 0, tzinfo=IST)) == expected


def test_import_all_moves_files_and_reports(monkeypatch):
    import sys
    import types

    from app.events import importer

    monkeypatch.setitem(sys.modules, "sqlalchemy", types.SimpleNamespace(text=lambda q: q))
    eng = _FakeEngine()
    files = {importer.PREFIX + "CF-AN-equities-RELIANCE.csv": SAMPLE.read_bytes(),
             importer.PREFIX + "notes.txt": b"x", importer.DONE + "old.csv": b"ignored"}
    moved = []
    st = importer.import_all(eng, lambda p: list(files), files.__getitem__, lambda a, b: moved.append((a, b)))
    assert st["files"] == 1 and st["announcements"] == 165 and st["results_meetings"] >= 4 and st["symbols"] == 1
    assert moved == [(importer.PREFIX + "CF-AN-equities-RELIANCE.csv", importer.DONE + "CF-AN-equities-RELIANCE.csv")]
    assert st["skipped_files"] == [importer.PREFIX + "notes.txt"]


def test_coverage_from_file_name_and_row_filter():
    from app.events import importer

    rows, _ = importer.parse_csv(SAMPLE.read_bytes())
    assert importer.coverage("imports/announcements/CF-AN-equities-RELIANCE-28-11-2025-to-28-11-2026.csv", rows) == \
        ("RELIANCE", date(2025, 11, 28), date(2026, 11, 28))
    assert importer.coverage("imports/announcements/my copy.csv", rows)[0] == "RELIANCE"   # falls back to rows
    df = pd.DataFrame({"symbol": ["RELIANCE", "RELIANCE", "TCS"],
                       "day": [date(2025, 11, 27), date(2026, 3, 2), date(2026, 3, 2)]})
    m = store.covered_by(df, [("RELIANCE", date(2025, 11, 28), date(2026, 11, 28))])
    assert m.tolist() == [False, True, False]
