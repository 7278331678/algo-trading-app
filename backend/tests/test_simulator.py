import pytest

from app.broker.base import BrokerError, OrderRequest, OrderStatus, OrderType, Side, new_order_tag
from app.broker.simulator import SimulatedBroker


def req(side=Side.BUY, qty=10, otype=OrderType.MARKET, price=None, trigger=None, tag=None, symbol="INFY"):
    return OrderRequest(
        symbol=symbol, side=side, quantity=qty, order_type=otype,
        price=price, trigger_price=trigger, tag=tag or new_order_tag(),
    )


@pytest.fixture
def broker():
    return SimulatedBroker(slippage_bps=0)


def test_tag_format():
    tag = new_order_tag()
    assert len(tag) == 20 and tag.isalnum()
    assert new_order_tag() != tag


def test_market_order_fills_on_next_price(broker):
    o = broker.place_order(req())
    assert o.status is OrderStatus.OPEN
    broker.on_price("INFY", 1500)
    assert o.status is OrderStatus.COMPLETE and o.average_price == 1500


def test_same_tag_never_places_twice(broker):
    r = req(tag="retry000000000000001")
    first = broker.place_order(r)
    second = broker.place_order(r)
    assert first.order_id == second.order_id
    assert len(broker.get_orders()) == 1


def test_limit_buy_waits_for_price(broker):
    broker.on_price("INFY", 1510)
    o = broker.place_order(req(otype=OrderType.LIMIT, price=1500))
    assert o.status is OrderStatus.OPEN
    broker.on_price("INFY", 1505)
    assert o.status is OrderStatus.OPEN
    broker.on_price("INFY", 1499)
    assert o.status is OrderStatus.COMPLETE and o.average_price == 1500  # resting limit fills at its price


def test_marketable_limit_fills_at_last_price(broker):
    broker.on_price("INFY", 1495)
    o = broker.place_order(req(otype=OrderType.LIMIT, price=1500))
    assert o.status is OrderStatus.COMPLETE and o.average_price == 1495


def test_stop_loss_protects_long_and_books_pnl(broker):
    broker.on_price("INFY", 1500)
    broker.place_order(req(qty=10))
    sl = broker.place_order(req(side=Side.SELL, qty=10, otype=OrderType.SL_M, trigger=1490))
    assert sl.status is OrderStatus.TRIGGER_PENDING
    broker.on_price("INFY", 1495)
    assert sl.status is OrderStatus.TRIGGER_PENDING
    broker.on_price("INFY", 1488)
    assert sl.status is OrderStatus.COMPLETE
    pos = broker.get_positions()["INFY"]
    assert pos.quantity == 0
    assert pos.realized_pnl == pytest.approx((1488 - 1500) * 10)


def test_slippage_worsens_market_fills():
    b = SimulatedBroker(slippage_bps=10)  # 0.10%
    b.on_price("INFY", 1000)
    buy = b.place_order(req())
    sell = b.place_order(req(side=Side.SELL))
    assert buy.average_price == 1001.0
    assert sell.average_price == 999.0


def test_cancel_and_cannot_cancel_twice(broker):
    o = broker.place_order(req(otype=OrderType.LIMIT, price=1))
    broker.cancel_order(o.order_id)
    assert o.status is OrderStatus.CANCELLED
    with pytest.raises(BrokerError):
        broker.cancel_order(o.order_id)


def test_square_off_flattens_everything(broker):
    broker.on_price("INFY", 1500)
    broker.on_price("TCS", 4000)
    broker.place_order(req(qty=5))
    broker.place_order(req(side=Side.SELL, qty=2, symbol="TCS"))
    resting = broker.place_order(req(otype=OrderType.LIMIT, price=1400))
    broker.on_price("INFY", 1510)
    broker.on_price("TCS", 3990)
    broker.square_off_all()
    positions = broker.get_positions()
    assert all(p.quantity == 0 for p in positions.values())
    assert resting.status is OrderStatus.CANCELLED
    assert positions["INFY"].realized_pnl == pytest.approx(50)
    assert positions["TCS"].realized_pnl == pytest.approx(20)


def test_max_order_value_rejects(broker):
    b = SimulatedBroker(max_order_value=10_000)
    o = b.place_order(req(qty=100, otype=OrderType.LIMIT, price=1500))
    assert o.status is OrderStatus.REJECTED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qty": 0},
        {"otype": OrderType.LIMIT},
        {"otype": OrderType.SL_M},
        {"tag": "bad tag!"},
    ],
)
def test_invalid_requests(broker, kwargs):
    with pytest.raises(ValueError):
        broker.place_order(req(**kwargs))


def test_stop_fills_at_trigger_when_crossed_continuously(broker):
    broker.on_price("INFY", 1500)
    broker.place_order(req(qty=10))
    sl = broker.place_order(req(side=Side.SELL, qty=10, otype=OrderType.SL_M, trigger=1490))
    broker.on_price("INFY", 1480, continuous=True)   # candle low well below the stop
    assert sl.average_price == 1490


def test_stop_fills_at_gap_price():
    b = SimulatedBroker(slippage_bps=0)
    b.on_price("INFY", 1500)
    b.place_order(req(qty=10))
    sl = b.place_order(req(side=Side.SELL, qty=10, otype=OrderType.SL_M, trigger=1490))
    b.on_price("INFY", 1470, continuous=False)       # next candle opens below the stop
    assert sl.average_price == 1470
