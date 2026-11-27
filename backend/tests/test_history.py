from datetime import date, datetime, timedelta

import pytest

from app.core.market_clock import IST
from app.marketdata import history


def ist(*a):
    return datetime(*a, tzinfo=IST)


def test_chunks_respect_kite_span_and_do_not_overlap():
    start, end = ist(2024, 11, 28), ist(2026, 11, 27, 16, 0)
    for interval, span in history.MAX_DAYS_PER_REQUEST.items():
        chunks = history.plan_chunks(start, end, interval)
        assert chunks[0][0] == start and chunks[-1][1] == end
        for a, b in chunks:
            assert (b.date() - a.date()).days + 1 <= span
        for (_, b), (c, _) in zip(chunks, chunks[1:], strict=False):
            assert c > b and c.date() == b.date() + timedelta(days=1)


def test_two_years_of_5min_is_8_requests():
    chunks = history.plan_chunks(ist(2024, 11, 28), ist(2026, 11, 27, 16), "5minute")
    assert len(chunks) == 8


def test_empty_when_start_after_end():
    assert history.plan_chunks(ist(2026, 1, 2), ist(2026, 1, 1), "day") == []


def test_unknown_interval():
    with pytest.raises(ValueError):
        history.plan_chunks(ist(2026, 1, 1), ist(2026, 1, 2), "3minute")


def test_resume_after_last_candle_or_window_start():
    window = date(2026, 1, 1)
    assert history.resume_start(None, window, "5minute") == ist(2026, 1, 1)
    last = ist(2026, 11, 26, 15, 25)
    assert history.resume_start(last, window, "5minute") == ist(2026, 11, 26, 15, 30)
    # stored data older than the window: restart at the window, not before
    assert history.resume_start(ist(2025, 6, 1), window, "day") == ist(2026, 1, 1)


def test_forming_candles_are_dropped():
    now = ist(2026, 11, 27, 10, 7)
    assert history.is_complete(ist(2026, 11, 27, 10, 0), "5minute", now)
    assert not history.is_complete(ist(2026, 11, 27, 10, 5), "5minute", now)
    assert not history.is_complete(ist(2026, 11, 27), "day", now)
    assert history.is_complete(ist(2026, 11, 27), "day", ist(2026, 11, 27, 15, 30))


def test_normalize_converts_and_filters():
    now = ist(2026, 11, 27, 10, 7)
    raw = [
        {"date": ist(2026, 11, 27, 10, 0), "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        {"date": ist(2026, 11, 27, 10, 5), "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
    ]
    out = history.normalize(raw, "5minute", now)
    assert len(out) == 1 and out[0]["volume"] == 10 and isinstance(out[0]["open"], float)


def test_rate_limiter_spaces_calls():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    rl = history.RateLimiter(2.0, clock=lambda: t[0], sleep=sleep)
    for _ in range(5):
        rl.wait()
    assert sum(slept) == pytest.approx(2.0)  # 5 calls at 2/s span 2 seconds


class NetworkException(Exception):
    pass


class TokenException(Exception):
    pass


class FakeKite:
    def __init__(self, failures):
        self.failures, self.calls = list(failures), 0

    def historical_data(self, *a):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return [{"date": ist(2026, 1, 1), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]


def _limiter():
    return history.RateLimiter(1000, clock=lambda: 0.0, sleep=lambda s: None)


def test_retries_transient_errors():
    kite = FakeKite([NetworkException("timeout"), NetworkException("timeout")])
    rows = history.fetch_chunk(kite, 1, ist(2026, 1, 1), ist(2026, 1, 2), "day", _limiter(), sleep=lambda s: None)
    assert len(rows) == 1 and kite.calls == 3


def test_token_error_aborts_immediately():
    kite = FakeKite([TokenException("expired")])
    with pytest.raises(history.TokenExpired):
        history.fetch_chunk(kite, 1, ist(2026, 1, 1), ist(2026, 1, 2), "day", _limiter(), sleep=lambda s: None)
    assert kite.calls == 1


def test_gives_up_after_retries():
    kite = FakeKite([NetworkException("x")] * 10)
    with pytest.raises(NetworkException):
        history.fetch_chunk(kite, 1, ist(2026, 1, 1), ist(2026, 1, 2), "day", _limiter(),
                            retries=2, sleep=lambda s: None)
    assert kite.calls == 3
