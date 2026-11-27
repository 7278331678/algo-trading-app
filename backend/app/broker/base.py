"""Broker interface shared by the real Kite adapter and the simulator.

Every caller (execution service, backtester, replay, tests) talks to this
interface only, so paper trading and live trading run the exact same code path.
"""

from __future__ import annotations

import secrets
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"        # stop-loss limit
    SL_M = "SL-M"    # stop-loss market


class OrderStatus(StrEnum):
    OPEN = "OPEN"                    # resting, waiting for price
    TRIGGER_PENDING = "TRIGGER PENDING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


TERMINAL_STATUSES = {OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.REJECTED}

_TAG_ALPHABET = string.ascii_letters + string.digits


def new_order_tag(prefix: str = "at") -> str:
    """Unique tag per order (Kite allows up to 20 alphanumeric characters).

    The tag is our idempotency key: placing the same tag twice must return the
    first order instead of creating a second one.
    """
    body = "".join(secrets.choice(_TAG_ALPHABET) for _ in range(20 - len(prefix)))
    return f"{prefix}{body}"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    quantity: int
    order_type: OrderType
    tag: str
    price: float | None = None           # LIMIT / SL
    trigger_price: float | None = None   # SL / SL-M
    exchange: str = "NSE"
    product: str = "MIS"                 # intraday

    def validate(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if not self.tag or len(self.tag) > 20 or not self.tag.isalnum():
            raise ValueError("tag must be 1-20 alphanumeric characters")
        if self.order_type in (OrderType.LIMIT, OrderType.SL) and not self.price:
            raise ValueError(f"{self.order_type.value} order needs a price")
        if self.order_type in (OrderType.SL, OrderType.SL_M) and not self.trigger_price:
            raise ValueError(f"{self.order_type.value} order needs a trigger price")


@dataclass
class Order:
    order_id: str
    request: OrderRequest
    status: OrderStatus
    placed_at: datetime
    filled_quantity: int = 0
    average_price: float = 0.0
    status_message: str = ""
    updated_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass
class Position:
    symbol: str
    quantity: int = 0            # >0 long, <0 short, 0 flat
    average_price: float = 0.0
    realized_pnl: float = 0.0
    fills: list[tuple[datetime, Side, int, float]] = field(default_factory=list)


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    """Minimal order-management surface the execution service needs."""

    name: str = "broker"

    @abstractmethod
    def place_order(self, request: OrderRequest) -> Order:
        """Place an order. Must be idempotent on request.tag."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order: ...

    @abstractmethod
    def get_order(self, order_id: str) -> Order: ...

    @abstractmethod
    def get_orders(self) -> list[Order]: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]: ...

    def find_by_tag(self, tag: str) -> Order | None:
        return next((o for o in self.get_orders() if o.request.tag == tag), None)
