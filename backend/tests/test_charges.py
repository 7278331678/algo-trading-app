import pytest

from app.trading.charges import breakeven_move_pct, order_brokerage, round_trip_charges


def test_brokerage_is_capped_at_20():
    assert order_brokerage(1_00_000) == 20
    assert order_brokerage(10_000) == pytest.approx(3.0)


def test_one_lakh_round_trip():
    c = round_trip_charges(1_00_000, 1_00_000)
    assert c.brokerage == 40
    assert c.stt == 25
    assert c.stamp == pytest.approx(3.0)
    assert 80 < c.total < 85  # about Rs 82, i.e. ~0.08% of trade value


def test_breakeven_is_about_eight_basis_points():
    assert 0.07 < breakeven_move_pct(1_00_000) < 0.09


def test_negative_values_rejected():
    with pytest.raises(ValueError):
        round_trip_charges(-1, 0)
