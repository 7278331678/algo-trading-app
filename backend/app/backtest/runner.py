"""Runs strategies x barrier settings on the stored feature files and saves the results.

Artifacts: backtests/<run_id>/<strategy>__<barrier>/{trades,daily}.parquet in object storage,
summary rows in the backtest_runs table.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, replace
from datetime import date

import pandas as pd

from app.core.market_clock import MarketClock
from app.features import build

from . import metrics, strategies
from .engine import BacktestConfig, run_backtest

log = logging.getLogger(__name__)

NEEDED = ["ts", "day", "symbol", "open", "high", "low", "close", "atr14", "daily_atr_pct", "exclude",
          "in_universe", "rvol", "adx14", "or_breakout", "vwap_dist", "ema50_15m_dist", "st_dir", "nifty_ret_open"]


def load_features(list_keys, get_bytes, start: date | None, end: date | None,
                  symbols: set[str] | None = None) -> pd.DataFrame:
    prefix = f"features/{build.FEATURE_VERSION}/5minute/"
    frames = []
    for key in sorted(list_keys(prefix)):
        sym = key.rsplit("/", 1)[-1].removesuffix(".parquet")
        if symbols and sym not in symbols:
            continue
        df = pd.read_parquet(io.BytesIO(get_bytes(key)), columns=NEEDED)
        df["day"] = pd.to_datetime(df["day"]).dt.date
        if start:
            df = df[df["day"] >= start]
        if end:
            df = df[df["day"] <= end]
        for c in df.select_dtypes("float64").columns:
            if c not in ("open", "high", "low", "close"):
                df[c] = df[c].astype("float32")
        frames.append(df)
    if not frames:
        raise RuntimeError("no feature files found: run build-features first")
    return pd.concat(frames, ignore_index=True)


def add_exit_times(df: pd.DataFrame, clock: MarketClock) -> pd.DataFrame:
    """Our square-off time per stock and day (auction stocks exit earlier; follows the data)."""
    last = df.groupby(["symbol", "day"])["ts"].max().dt.time
    exit_map = {}
    for (sym, d), last_t in last.items():
        cas_end = clock.rules.continuous_end(d, cas=True)
        cas = cas_end < clock.rules.continuous_end(d, cas=False) and last_t < cas_end
        exit_map[(sym, d)] = clock.square_off_time(d, cas=cas)
    keys = list(zip(df["symbol"], df["day"], strict=True))
    df = df.copy()
    df["exit_time"] = [exit_map[k] for k in keys]
    return df


def run_study(df: pd.DataFrame, base: BacktestConfig, strategy_names: list[str], grid: list[tuple]) -> list[dict]:
    results = []
    for sname in strategy_names:
        signals = strategies.STRATEGIES[sname](df)
        for gname, basis, sm, tm in grid:
            # research runs measure the whole period (no halt) and report when the kill switch WOULD trip
            cfg = replace(base, stop_basis=basis, stop_mult=sm, target_mult=tm, max_drawdown_halt_pct=1e9)
            res = run_backtest(signals, cfg)
            summary = metrics.summarize(res.trades, res.daily, cfg.capital)
            summary["would_halt_on"] = metrics.halt_day(res.daily, base.max_drawdown_halt_pct)
            periods = metrics.by_period(res.trades, res.daily, cfg.capital)
            summary["profitable_quarters"] = metrics.consistency(periods)
            log.info("%s / %s: %s trades, net %s, PF %s", sname, gname, summary.get("trades"),
                     summary.get("net_pnl"), summary.get("profit_factor"))
            results.append({"name": f"{sname}__{gname}", "strategy": sname, "barrier": gname,
                            "config": {k: str(v) for k, v in asdict(cfg).items()},
                            "summary": summary, "periods": periods, "trades": res.trades, "daily": res.daily})
    return results


def save_results(engine, put_bytes, run_id: str, results: list[dict], period: dict) -> list[int]:
    from sqlalchemy import text

    ids = []
    for r in results:
        prefix = f"backtests/{run_id}/{r['name']}"
        for part in ("trades", "daily"):
            buf = io.BytesIO()
            frame = r[part].copy()
            for c in frame.columns:
                if frame[c].dtype == object:
                    frame[c] = frame[c].astype(str)
            frame.to_parquet(buf, index=False)
            put_bytes(f"{prefix}/{part}.parquet", buf.getvalue())
        with engine.begin() as conn:
            ids.append(conn.execute(text(
                "INSERT INTO backtest_runs (run_id, name, strategy, barrier, params, summary, periods, artifacts)"
                " VALUES (:rid, :n, :s, :b, CAST(:p AS jsonb), CAST(:su AS jsonb), CAST(:pe AS jsonb), :a)"
                " RETURNING id"),
                {"rid": run_id, "n": r["name"], "s": r["strategy"], "b": r["barrier"],
                 "p": json.dumps({**r["config"], **period}), "su": json.dumps(r["summary"], default=str),
                 "pe": json.dumps(r["periods"], default=str), "a": prefix}).scalar_one())
    return ids
