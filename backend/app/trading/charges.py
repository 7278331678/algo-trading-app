"""Zerodha equity-intraday charges.

Rates come from zerodha.com/charges and are kept in one place so the backtester,
the simulator and the reports all compute the same cost. Check them against the
charges page whenever Zerodha or the exchange revises a rate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntradayRates:
    brokerage_pct: float = 0.0003          # 0.03% per executed order...
    brokerage_cap: float = 20.0            # ...or Rs 20, whichever is lower
    stt_sell_pct: float = 0.00025          # 0.025% on the sell side
    exchange_txn_pct: float = 0.0000307    # NSE transaction charge (check periodically)
    sebi_per_crore: float = 10.0           # Rs 10 per crore of turnover
    stamp_buy_pct: float = 0.00003         # 0.003% on the buy side
    gst_pct: float = 0.18                  # on brokerage + exchange + SEBI charges


DEFAULT_RATES = IntradayRates()


@dataclass(frozen=True)
class ChargeBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst


def order_brokerage(value: float, rates: IntradayRates = DEFAULT_RATES) -> float:
    return min(value * rates.brokerage_pct, rates.brokerage_cap)


def round_trip_charges(
    buy_value: float, sell_value: float, rates: IntradayRates = DEFAULT_RATES
) -> ChargeBreakdown:
    """Charges for one intraday round trip (one buy order and one sell order)."""
    if buy_value < 0 or sell_value < 0:
        raise ValueError("trade values must be non-negative")
    turnover = buy_value + sell_value
    brokerage = order_brokerage(buy_value, rates) + order_brokerage(sell_value, rates)
    stt = round(sell_value * rates.stt_sell_pct)  # Zerodha rounds STT to the rupee
    exchange = turnover * rates.exchange_txn_pct
    sebi = turnover * rates.sebi_per_crore / 1e7
    stamp = buy_value * rates.stamp_buy_pct
    gst = (brokerage + exchange + sebi) * rates.gst_pct
    return ChargeBreakdown(brokerage, stt, exchange, sebi, stamp, gst)


def breakeven_move_pct(trade_value: float, rates: IntradayRates = DEFAULT_RATES) -> float:
    """Price move (in %) a trade needs just to cover its charges."""
    return round_trip_charges(trade_value, trade_value, rates).total / trade_value * 100
