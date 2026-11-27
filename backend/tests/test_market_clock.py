import json
from datetime import UTC, date, datetime

import pytest

from app.core.market_clock import IST, HolidayCalendarMissing, MarketClock, Phase


@pytest.fixture
def clock(tmp_path):
    cal = {
        "years": {
            "2026": {
                "holidays": [{"date": "2026-12-25", "name": "Christmas"}],
                "special_sessions": [
                    {"date": "2026-11-08", "start": "18:00", "end": "19:00", "name": "Muhurat"}
                ],
            }
        }
    }
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(cal))
    return MarketClock(path)


def ist(*args):
    return datetime(*args, tzinfo=IST)


def test_weekend_and_holiday(clock):
    assert not clock.is_trading_day(date(2026, 11, 28))  # Saturday
    assert not clock.is_trading_day(date(2026, 12, 25))  # holiday
    assert clock.is_trading_day(date(2026, 11, 27))      # Friday


@pytest.mark.parametrize(
    "hh,mm,expected",
    [
        (8, 30, Phase.CLOSED),
        (9, 5, Phase.PRE_OPEN),
        (9, 20, Phase.OPENING_RANGE),
        (9, 30, Phase.ENTRY_WINDOW),
        (14, 29, Phase.ENTRY_WINDOW),
        (14, 30, Phase.MANAGE_ONLY),
        (15, 15, Phase.SQUARE_OFF),
        (15, 31, Phase.POST_CLOSE),
    ],
)
def test_phases(clock, hh, mm, expected):
    assert clock.phase(ist(2026, 11, 27, hh, mm)) is expected


def test_entries_only_in_window(clock):
    assert clock.can_open_new_trades(ist(2026, 11, 27, 10, 0))
    assert not clock.can_open_new_trades(ist(2026, 11, 27, 9, 20))
    assert not clock.can_open_new_trades(ist(2026, 12, 25, 10, 0))


def test_missing_year_refuses_to_guess(clock):
    with pytest.raises(HolidayCalendarMissing):
        clock.is_trading_day(date(2027, 1, 4))


def test_naive_datetime_rejected(clock):
    with pytest.raises(ValueError):
        clock.phase(datetime(2026, 11, 27, 10, 0))


def test_utc_input_converted(clock):

    # 04:30 UTC == 10:00 IST
    assert clock.phase(datetime(2026, 11, 27, 4, 30, tzinfo=UTC)) is Phase.ENTRY_WINDOW


def test_kite_token_expiry(clock):
    assert clock.kite_token_expiry(ist(2026, 11, 27, 8, 50)) == ist(2026, 11, 28, 6, 0)
    assert clock.kite_token_expiry(ist(2026, 11, 27, 5, 0)) == ist(2026, 11, 27, 6, 0)


def test_next_trading_day_skips_weekend_and_holiday(clock):
    assert clock.next_trading_day(date(2026, 11, 27)) == date(2026, 11, 30)
    assert clock.next_trading_day(date(2026, 12, 24)) == date(2026, 12, 28)


def test_special_session(clock):
    s = clock.special_session(date(2026, 11, 8))
    assert s and s.name == "Muhurat"


def test_bundled_calendar_loads():
    MarketClock()  # the shipped JSON must parse
