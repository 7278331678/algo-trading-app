"""Broker simulator: paper trading, backtests, market replay and tests.

Zerodha has no paper-trading API, so this class stands in for Kite behind the
same Broker interface. Feed it prices with on_price(); it fills resting orders
the way an exchange would, with configurable slippage.

Fill rules (deliberately conservative):
- MARKET: fills at the last price, worsened by slippage.
- LIMIT: a resting limit fills at its own price once the market trades at or
  through it (fed candle highs/lows must not hand it a better price); a limit
  that is marketable when placed fills at the last price.
- SL-M: triggers when the price touches the trigger. If the price moved there
  continuously it fills at the trigger (+ slippage); if it jumped past it (a gap,
  continuous=False, e.g. a candle open) it fills at that price (+ slippage).
- SL: triggers like SL-M, then behaves as a LIMIT at its price.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import replace
from datetime import datetime

from app.core.market_clock import IST

from .base import (
    Broker,
    BrokerError,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Side,
)


class SimulatedBroker(Broker):
    name = "simulator"

    def __init__(self, slippage_bps: float = 5.0, max_order_value: float | None = None) -> None:
        self.slippage_bps = slippage_bps
        self.max_order_value = max_order_value
        self._orders: dict[str, Order] = {}
        self._by_tag: dict[str, str] = {}
        self._positions: dict[str, Position] = {}
        self._last_price: dict[str, float] = {}
        self._last_ts: datetime | None = None
        self._ids = itertools.count(1)
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- helpers
    def _now(self) -> datetime:
        return self._last_ts or datetime.now(IST)

    def _slip(self, price: float, side: Side) -> float:
        adj = price * self.slippage_bps / 10_000
        return round(price + adj if side is Side.BUY else price - adj, 2)

    def _fill(self, order: Order, price: float) -> None:
        req = order.request
        order.status = OrderStatus.COMPLETE
        order.filled_quantity = req.quantity
        order.average_price = price
        order.updated_at = self._now()
        self._apply_fill(req.symbol, req.side, req.quantity, price)

    def _apply_fill(self, symbol: str, side: Side, qty: int, price: float) -> None:
        pos = self._positions.setdefault(symbol, Position(symbol=symbol))
        signed = qty if side is Side.BUY else -qty
        pos.fills.append((self._now(), side, qty, price))
        if pos.quantity == 0 or (pos.quantity > 0) == (signed > 0):
            # opening or adding: weighted average price
            total = abs(pos.quantity) + qty
            pos.average_price = (pos.average_price * abs(pos.quantity) + price * qty) / total
            pos.quantity += signed
            return
        # reducing or flipping
        closing = min(qty, abs(pos.quantity))
        direction = 1 if pos.quantity > 0 else -1
        pos.realized_pnl += (price - pos.average_price) * closing * direction
        pos.quantity += signed
        if pos.quantity == 0:
            pos.average_price = 0.0
        elif (pos.quantity > 0) != (direction > 0):
            pos.average_price = price  # flipped: remainder opened at this price

    def _try_execute(self, order: Order, price: float, resting: bool = True, continuous: bool = False) -> None:
        req = order.request
        if order.status is OrderStatus.TRIGGER_PENDING:
            hit = price <= req.trigger_price if req.side is Side.SELL else price >= req.trigger_price
            if not hit:
                return
            if req.order_type is OrderType.SL_M:
                ref = req.trigger_price if (continuous and resting) else price
                self._fill(order, self._slip(ref, req.side))
                return
            order.status = OrderStatus.OPEN  # SL becomes a resting limit
        if order.status is not OrderStatus.OPEN:
            return
        if req.order_type in (OrderType.MARKET, OrderType.SL_M):
            self._fill(order, self._slip(price, req.side))
        elif req.side is Side.BUY and price <= req.price:
            self._fill(order, req.price if resting else price)
        elif req.side is Side.SELL and price >= req.price:
            self._fill(order, req.price if resting else price)

    # -------------------------------------------------------------- Broker API
    def place_order(self, request: OrderRequest) -> Order:
        request.validate()
        with self._lock:
            if request.tag in self._by_tag:  # idempotent retry
                return self._orders[self._by_tag[request.tag]]
            order_id = f"SIM{next(self._ids):08d}"
            status = (
                OrderStatus.TRIGGER_PENDING
                if request.order_type in (OrderType.SL, OrderType.SL_M)
                else OrderStatus.OPEN
            )
            order = Order(order_id=order_id, request=request, status=status, placed_at=self._now())
            self._orders[order_id] = order
            self._by_tag[request.tag] = order_id

            ref = request.price or self._last_price.get(request.symbol)
            if self.max_order_value and ref and ref * request.quantity > self.max_order_value:
                order.status = OrderStatus.REJECTED
                order.status_message = "order value above max_order_value"
                return order

            last = self._last_price.get(request.symbol)
            if last is not None:
                self._try_execute(order, last, resting=False)
            return order

    def cancel_order(self, order_id: str) -> Order:
        with self._lock:
            order = self.get_order(order_id)
            if order.is_terminal:
                raise BrokerError(f"order {order_id} is already {order.status.value}")
            order.status = OrderStatus.CANCELLED
            order.updated_at = self._now()
            return order

    def get_order(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise BrokerError(f"unknown order {order_id}") from exc

    def get_orders(self) -> list[Order]:
        with self._lock:
            return list(self._orders.values())

    def get_positions(self) -> dict[str, Position]:
        with self._lock:
            return {s: replace(p, fills=list(p.fills)) for s, p in self._positions.items()}

    def find_by_tag(self, tag: str) -> Order | None:
        oid = self._by_tag.get(tag)
        return self._orders.get(oid) if oid else None

    # -------------------------------------------------------- simulation feed
    def on_price(self, symbol: str, price: float, ts: datetime | None = None,
                 continuous: bool = False) -> list[Order]:
        """Feed a new traded price; returns orders that filled on it.

        continuous=True means the market moved here smoothly from the previous price (e.g. a
        candle's low after its open), so a stop crossed on the way fills at its trigger."""
        if price <= 0:
            raise ValueError("price must be positive")
        with self._lock:
            if ts is not None:
                self._last_ts = ts
            self._last_price[symbol] = price
            filled = []
            for order in self._orders.values():
                if order.request.symbol != symbol or order.is_terminal:
                    continue
                self._try_execute(order, price, resting=True, continuous=continuous)
                if order.status is OrderStatus.COMPLETE:
                    filled.append(order)
            return filled

    def square_off_all(self) -> None:
        """Cancel every open order and close every position at the last price."""
        with self._lock:
            for order in self._orders.values():
                if not order.is_terminal:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = self._now()
            for symbol, pos in self._positions.items():
                if pos.quantity == 0:
                    continue
                last = self._last_price.get(symbol)
                if last is None:
                    raise BrokerError(f"no price for {symbol}; cannot square off")
                side = Side.SELL if pos.quantity > 0 else Side.BUY
                self._apply_fill(symbol, side, abs(pos.quantity), self._slip(last, side))
