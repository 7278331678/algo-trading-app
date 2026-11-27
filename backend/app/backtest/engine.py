"""Day-by-day backtest on 5-minute candles, executed through the broker simulator.

Per stock and candle the price path fed to the simulator is open -> adverse extreme ->
favourable extreme -> close (for a long: open, low, high, close), so when a candle touches
both the stop and the target the stop fills first (conservative, same as the labels).

Flow for a signal on candle t (decided when t closes):
  1. market entry at the open of the stock's next candle (+ slippage)
  2. immediately a stop-loss (SL-M) and a profit target (LIMIT) are placed; when one fills
     the other is cancelled (OCO)
  3. anything still open is closed at the close of the last candle ending by our square-off time
Costs: Zerodha intraday charges on every round trip (app.trading.charges).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from app.broker.base import OrderRequest, OrderStatus, OrderType, Side, new_order_tag
from app.broker.simulator import SimulatedBroker
from app.trading.charges import round_trip_charges


@dataclass(frozen=True)
class BacktestConfig:
    capital: float = 100_000.0
    risk_pct: float = 0.5                 # % of capital lost if the stop is hit
    max_notional_per_trade: float = 1.0   # x capital
    max_total_exposure: float = 5.0       # x capital (MIS leverage)
    max_trades_per_day: int = 5
    daily_loss_cap_pct: float = 2.0       # stop opening trades once the day's realised loss reaches this
    max_drawdown_halt_pct: float = 15.0   # kill switch: stop trading for good once equity is this far below its peak
    stop_basis: str = "atr5"              # "atr5" = 5-min ATR(14); "daily" = previous day's daily ATR(14)
    stop_mult: float = 1.0
    target_mult: float = 1.5
    slippage_bps: float = 5.0
    entry_start: time = time(9, 30)
    entry_end: time = time(14, 30)
    bar_minutes: int = 5

    def unit(self, row) -> float:
        """Distance of 1 'stop unit' in rupees for this row."""
        if self.stop_basis == "atr5":
            return float(row.atr14)
        if self.stop_basis == "daily":
            return float(row.daily_atr_pct) * float(row.close)
        raise ValueError(f"unknown stop_basis {self.stop_basis}")


@dataclass
class Trade:
    day: object
    symbol: str
    side: str
    signal_time: datetime
    stop_dist: float
    target_dist: float
    qty: int
    entry_time: datetime | None = None
    entry_price: float = float("nan")
    exit_time: datetime | None = None
    exit_price: float = float("nan")
    exit_reason: str = ""
    gross: float = 0.0                    # after slippage, before charges
    slippage: float = 0.0                 # rupees lost to slippage on entry and exit
    charges: float = 0.0
    net: float = 0.0
    r_multiple: float = float("nan")
    orders: dict = field(default_factory=dict)  # role -> order_id

    def record(self) -> dict:
        d = asdict(self)
        d.pop("orders")
        return d


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    config: BacktestConfig
    halted_on: object = None              # day the kill switch tripped, if it did


REQUIRED = ["ts", "day", "symbol", "open", "high", "low", "close", "atr14", "daily_atr_pct",
            "exit_time", "signal", "score"]


def run_backtest(bars: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    missing = set(REQUIRED) - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    bars = bars.sort_values(["day", "ts", "symbol"])
    trades: list[Trade] = []
    daily_rows = []
    equity = peak = cfg.capital
    halted_on = None
    for day, dframe in bars.groupby("day", sort=True):
        if halted_on is None and equity <= peak * (1 - cfg.max_drawdown_halt_pct / 100):
            halted_on = day
        if halted_on is not None:   # kill switch: no more trading, like the live risk agent
            daily_rows.append({"day": day, "trades": 0, "gross": 0.0, "charges": 0.0, "net": 0.0,
                               "equity": equity})
            continue
        day_trades = _run_day(day, dframe, cfg)
        trades.extend(day_trades)
        gross = sum(t.gross for t in day_trades)
        charges = sum(t.charges for t in day_trades)
        equity += gross - charges
        peak = max(peak, equity)
        daily_rows.append({"day": day, "trades": len(day_trades), "gross": gross, "charges": charges,
                           "net": gross - charges, "equity": equity})
    tdf = pd.DataFrame([t.record() for t in trades])
    return BacktestResult(trades=tdf, daily=pd.DataFrame(daily_rows), config=cfg, halted_on=halted_on)


def _run_day(day, dframe: pd.DataFrame, cfg: BacktestConfig) -> list[Trade]:
    broker = SimulatedBroker(slippage_bps=cfg.slippage_bps)
    step = timedelta(minutes=cfg.bar_minutes)
    open_trades: dict[str, Trade] = {}
    pending: dict[str, Trade] = {}
    done: list[Trade] = []
    traded_symbols: set[str] = set()
    realised = 0.0
    loss_cap = cfg.capital * cfg.daily_loss_cap_pct / 100

    def close_trade(tr: Trade, price: float, when, reason: str, ref: float) -> None:
        nonlocal realised
        tr.exit_price, tr.exit_time, tr.exit_reason = price, when, reason
        tr.slippage += abs(price - ref) * tr.qty   # ref = price before slippage
        sign = 1 if tr.side == "long" else -1
        tr.gross = (tr.exit_price - tr.entry_price) * tr.qty * sign
        buy, sell = (tr.entry_price, tr.exit_price) if sign == 1 else (tr.exit_price, tr.entry_price)
        tr.charges = round_trip_charges(buy * tr.qty, sell * tr.qty).total
        tr.net = tr.gross - tr.charges
        tr.r_multiple = tr.net / (tr.stop_dist * tr.qty) if tr.stop_dist > 0 else float("nan")
        realised += tr.net
        done.append(tr)
        open_trades.pop(tr.symbol, None)

    # plain arrays for speed: the inner loop runs for every candle of every day
    sym_a = dframe["symbol"].to_numpy()
    ts_a = dframe["ts"].to_numpy()
    o_a, h_a = dframe["open"].to_numpy(float), dframe["high"].to_numpy(float)
    l_a, c_a = dframe["low"].to_numpy(float), dframe["close"].to_numpy(float)
    atr_a, datr_a = dframe["atr14"].to_numpy(float), dframe["daily_atr_pct"].to_numpy(float)
    ex_a = dframe["exit_time"].to_numpy()
    sig_a, score_a = dframe["signal"].to_numpy(), dframe["score"].to_numpy(float)
    ts_list = list(dframe["ts"])
    bounds = np.flatnonzero(np.r_[True, ts_a[1:] != ts_a[:-1], True])

    class Row:  # the fields the logic below reads, for one stock at one candle
        __slots__ = ("symbol", "open", "high", "low", "close", "atr14", "daily_atr_pct", "exit_time",
                     "signal", "score")

        def __init__(self, i):
            self.symbol, self.open, self.high, self.low, self.close = sym_a[i], o_a[i], h_a[i], l_a[i], c_a[i]
            self.atr14, self.daily_atr_pct, self.exit_time = atr_a[i], datr_a[i], ex_a[i]
            self.signal, self.score = sig_a[i], score_a[i]

    for bi in range(len(bounds) - 1):
        a, b = bounds[bi], bounds[bi + 1]
        ts = ts_list[a]
        bar_end = ts + step
        active = set(pending) | set(open_trades)
        sig_idx = a + np.flatnonzero(sig_a[a:b] != 0)
        if not active and len(sig_idx) == 0:
            continue
        rows = {}
        if active:
            for i in range(a, b):
                if sym_a[i] in active:
                    rows[sym_a[i]] = Row(i)
        for i in sig_idx:
            rows.setdefault(sym_a[i], Row(i))

        # 1. move every stock that has a pending entry or an open trade through this candle
        for sym in list(pending) + list(open_trades):
            r = rows.get(sym)
            if r is None:
                continue
            tr = pending.get(sym) or open_trades.get(sym)
            long = tr.side == "long"
            exit_t = r.exit_time
            if sym in pending and bar_end.time() > exit_t:
                pending.pop(sym)          # entry candle would end after our exit time: skip the trade
                continue
            path = [r.open, r.low, r.high, r.close] if long else [r.open, r.high, r.low, r.close]
            for k, px in enumerate(path):
                # the open may gap past a stop (fills at the open); later points are continuous moves
                broker.on_price(sym, float(px), ts if k == 0 else ts + step / 2, continuous=k > 0)
                if k == 0 and sym in pending:
                    tr = pending.pop(sym)
                    side = Side.BUY if long else Side.SELL
                    o = broker.place_order(OrderRequest(sym, side, tr.qty, OrderType.MARKET, new_order_tag("en")))
                    tr.entry_price, tr.entry_time = o.average_price, ts
                    tr.slippage = abs(o.average_price - float(px)) * tr.qty
                    exit_side = Side.SELL if long else Side.BUY
                    stop_px = round(tr.entry_price - tr.stop_dist if long else tr.entry_price + tr.stop_dist, 2)
                    tgt_px = round(tr.entry_price + tr.target_dist if long else tr.entry_price - tr.target_dist, 2)
                    tr.orders["stop"] = broker.place_order(OrderRequest(
                        sym, exit_side, tr.qty, OrderType.SL_M, new_order_tag("sl"), trigger_price=stop_px)).order_id
                    tr.orders["target"] = broker.place_order(OrderRequest(
                        sym, exit_side, tr.qty, OrderType.LIMIT, new_order_tag("tg"), price=tgt_px)).order_id
                    open_trades[sym] = tr
                    continue
                if sym in open_trades:
                    stop_o = broker.get_order(tr.orders["stop"])
                    tgt_o = broker.get_order(tr.orders["target"])
                    if stop_o.status is OrderStatus.COMPLETE:
                        broker.cancel_order(tgt_o.order_id)
                        # reference: the trigger if crossed smoothly, the open if it gapped past it
                        ref = stop_o.request.trigger_price if k > 0 else float(px)
                        close_trade(tr, stop_o.average_price, ts, "stop", ref)
                        break
                    if tgt_o.status is OrderStatus.COMPLETE:
                        broker.cancel_order(stop_o.order_id)
                        close_trade(tr, tgt_o.average_price, ts, "target", tgt_o.average_price)
                        break
            # 2. square off at the close of the last candle ending by our exit time
            if sym in open_trades and bar_end.time() >= exit_t:
                tr = open_trades[sym]
                for role in ("stop", "target"):
                    if not broker.get_order(tr.orders[role]).is_terminal:
                        broker.cancel_order(tr.orders[role])
                side = Side.SELL if tr.side == "long" else Side.BUY
                o = broker.place_order(OrderRequest(sym, side, tr.qty, OrderType.MARKET, new_order_tag("sq")))
                close_trade(tr, o.average_price, bar_end, "time", float(r.close))

        # 3. new entries decided at this candle's close
        dec = bar_end.time()
        if not (cfg.entry_start <= dec <= cfg.entry_end):
            continue
        if realised <= -loss_cap or len(traded_symbols) >= cfg.max_trades_per_day:
            continue
        cands = sorted((rows[sym_a[i]] for i in sig_idx), key=lambda r: -r.score)
        for r in cands:
            if len(traded_symbols) >= cfg.max_trades_per_day:
                break
            if r.symbol in traded_symbols or dec >= r.exit_time:
                continue
            unit = cfg.unit(r)
            if not np.isfinite(unit) or unit <= 0:
                continue
            stop_dist, target_dist = cfg.stop_mult * unit, cfg.target_mult * unit
            qty = int(cfg.capital * cfg.risk_pct / 100 // stop_dist)
            qty = min(qty, int(cfg.capital * cfg.max_notional_per_trade // r.close))
            exposure = sum(t.qty * t.entry_price for t in open_trades.values()) + \
                sum(t.qty * float(rows[s].close) if s in rows else 0 for s, t in pending.items())
            room = cfg.capital * cfg.max_total_exposure - exposure
            qty = min(qty, int(max(room, 0) // r.close))
            if qty <= 0:
                continue
            pending[r.symbol] = Trade(day=day, symbol=r.symbol, side="long" if r.signal > 0 else "short",
                                      signal_time=bar_end, stop_dist=stop_dist, target_dist=target_dist, qty=qty)
            traded_symbols.add(r.symbol)

    # safety net: nothing may stay open overnight (should not trigger: step 2 squares off)
    for sym, tr in list(open_trades.items()):
        last = dframe[dframe["symbol"] == sym].iloc[-1]
        close_trade(tr, float(last["close"]), last["ts"] + step, "eod", float(last["close"]))
    return done
