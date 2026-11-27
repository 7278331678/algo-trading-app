"""Zerodha Kite Connect adapter behind the Broker interface.

Orders from this adapter only reach the exchange when TRADING_MODE=live and the
request comes from the whitelisted static IP. The execution service is the only
caller; no LLM or agent ever imports this module.
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.core.market_clock import IST

from .base import Broker, BrokerError, Order, OrderRequest, OrderStatus, OrderType, Position, Side

log = logging.getLogger(__name__)

_STATUS_MAP = {
    "COMPLETE": OrderStatus.COMPLETE,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "TRIGGER PENDING": OrderStatus.TRIGGER_PENDING,
}


class KiteBroker(Broker):
    name = "kite"

    def __init__(self, api_key: str, access_token: str, kite_client=None) -> None:
        if kite_client is None:
            from kiteconnect import KiteConnect  # imported lazily: not needed in tests

            kite_client = KiteConnect(api_key=api_key)
            kite_client.set_access_token(access_token)
        self.kite = kite_client

    # ---------------------------------------------------------------- mapping
    @staticmethod
    def _to_order(raw: dict) -> Order:
        req = OrderRequest(
            symbol=raw["tradingsymbol"],
            side=Side(raw["transaction_type"]),
            quantity=int(raw["quantity"]),
            order_type=OrderType(raw["order_type"]),
            tag=raw.get("tag") or "untagged",
            price=raw.get("price") or None,
            trigger_price=raw.get("trigger_price") or None,
            exchange=raw.get("exchange", "NSE"),
            product=raw.get("product", "MIS"),
        )
        placed = raw.get("order_timestamp") or datetime.now(IST)
        if isinstance(placed, datetime) and placed.tzinfo is None:
            placed = placed.replace(tzinfo=IST)
        return Order(
            order_id=str(raw["order_id"]),
            request=req,
            # Kite has several transient states (VALIDATION PENDING, OPEN PENDING...): treat as OPEN
            status=_STATUS_MAP.get(raw.get("status", ""), OrderStatus.OPEN),
            placed_at=placed,
            filled_quantity=int(raw.get("filled_quantity") or 0),
            average_price=float(raw.get("average_price") or 0.0),
            status_message=raw.get("status_message") or "",
        )

    # -------------------------------------------------------------- Broker API
    def place_order(self, request: OrderRequest) -> Order:
        request.validate()
        existing = self.find_by_tag(request.tag)
        if existing:  # idempotent: a retry never places a second order
            log.warning("order with tag %s already exists (%s)", request.tag, existing.order_id)
            return existing
        try:
            order_id = self.kite.place_order(
                variety="regular",
                exchange=request.exchange,
                tradingsymbol=request.symbol,
                transaction_type=request.side.value,
                quantity=request.quantity,
                product=request.product,
                order_type=request.order_type.value,
                price=request.price,
                trigger_price=request.trigger_price,
                tag=request.tag,
            )
        except Exception as exc:
            # The request may have reached Zerodha before the error: check by tag.
            landed = self.find_by_tag(request.tag)
            if landed:
                log.warning("place_order raised but order %s exists; using it", landed.order_id)
                return landed
            raise BrokerError(f"place_order failed for {request.symbol}: {exc}") from exc
        return self.get_order(str(order_id))

    def cancel_order(self, order_id: str) -> Order:
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
        except Exception as exc:
            raise BrokerError(f"cancel_order failed for {order_id}: {exc}") from exc
        return self.get_order(order_id)

    def get_order(self, order_id: str) -> Order:
        for order in self.get_orders():
            if order.order_id == order_id:
                return order
        raise BrokerError(f"unknown order {order_id}")

    def get_orders(self) -> list[Order]:
        return [self._to_order(o) for o in self.kite.orders()]

    def get_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self.kite.positions().get("day", []):
            if p.get("product") != "MIS":
                continue
            out[p["tradingsymbol"]] = Position(
                symbol=p["tradingsymbol"],
                quantity=int(p["quantity"]),
                average_price=float(p.get("average_price") or 0.0),
                realized_pnl=float(p.get("realised") or 0.0),
            )
        return out
