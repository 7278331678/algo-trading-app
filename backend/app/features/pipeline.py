"""Builds feature + label frames for the whole universe and stores them as Parquet.

Output (object storage: MinIO locally, S3 in cloud):
  features/<version>/5minute/<SYMBOL>.parquet   one file per stock
  features/<version>/manifest.json              feature list, parameters, coverage, code hash
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from app.core.market_clock import IST, MarketClock
from app.marketdata import store, universe

from . import build, exclusions, labels

log = logging.getLogger(__name__)

LABEL_PARAMS = {"target_mult": 1.5, "stop_mult": 1.0, "entry_start": "09:30", "entry_end": "14:30",
                "entry": "next candle open", "ties": "stop first", "exit": "our square-off (session_rules.json)"}


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(p.read_bytes())
    for name in ("session_rules.json", "corporate_actions.csv"):
        h.update((Path(__file__).resolve().parent.parent / "data" / name).read_bytes())
    return h.hexdigest()[:16]


def _load(engine, token: int, interval: str, since: date) -> pd.DataFrame:
    rows = store.load_candles(engine, token, interval, datetime.combine(since, datetime.min.time(), tzinfo=IST))
    return build.frame(rows)


def _exit_times(m5: pd.DataFrame, clock: MarketClock) -> dict:
    """Our square-off time per day. On auction days the stock is treated as an auction (F&O)
    stock if its candles stop before the normal close; this follows the data, not today's list."""
    out = {}
    last = m5.groupby("day")["ts"].max().dt.time
    for d, last_t in last.items():
        cas_end = clock.rules.continuous_end(d, cas=True)
        normal_end = clock.rules.continuous_end(d, cas=False)
        cas = cas_end < normal_end and last_t < cas_end
        out[d] = clock.square_off_time(d, cas=cas)
    return out


def _quality_jumps(engine) -> dict[str, set[date]]:
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT i.tradingsymbol, q.day FROM data_quality_issues q JOIN instruments i USING (instrument_token)"
            " WHERE q.issue='price_jump' AND q.run_id = (SELECT max(run_id) FROM data_quality_issues)"
        )).all()
    out: dict[str, set[date]] = {}
    for sym, d in rows:
        out.setdefault(sym, set()).add(d)
    return out


def build_universe_features(settings, engine, clock: MarketClock, put_bytes, symbols: set[str] | None = None,
                            now: datetime | None = None) -> dict:
    from sqlalchemy import text

    now = now or clock.now()
    today = now.date()
    since = today - timedelta(days=settings.retention_5minute_days)
    daily_since = today - timedelta(days=settings.retention_day_days)
    targets = universe.tokens_to_fetch(engine, settings.universe_index, since)

    with engine.connect() as conn:
        idx = {s: int(t) for t, s in conn.execute(text(
            "SELECT instrument_token, tradingsymbol FROM instruments WHERE segment='INDICES'"))}
        spells = conn.execute(text(
            "SELECT symbol, sector_index, valid_from, valid_to FROM universe_members WHERE index_name=:i"),
            {"i": settings.universe_index}).all()
    stocks = {t: s for t, s in targets.items() if t not in set(idx.values())}
    if symbols:
        stocks = {t: s for t, s in stocks.items() if s in symbols}
    if universe.BROAD_INDEX not in idx:
        raise RuntimeError("NIFTY 50 token missing: run universe-sync")
    sector_of = {s: sec for s, sec, _, vt in spells if vt is None}

    # market context
    nifty_m5 = _load(engine, idx[universe.BROAD_INDEX], "5minute", since)
    nifty_ctx = build.index_context(nifty_m5, _load(engine, idx[universe.BROAD_INDEX], "15minute", since), "nifty")
    special = {d for d, n in nifty_m5.groupby("day").size().items() if n < 0.75 * 75}
    vix_ctx = None
    if "INDIA VIX" in idx:
        vix_ctx = build.vix_context(_load(engine, idx["INDIA VIX"], "5minute", since),
                                    _load(engine, idx["INDIA VIX"], "day", daily_since))
    sector_ctx: dict[str, pd.DataFrame] = {}
    for sec in sorted({s for s in sector_of.values() if s and s in idx}):
        sector_ctx[sec] = build.index_context(_load(engine, idx[sec], "5minute", since),
                                              _load(engine, idx[sec], "15minute", since), "sector")

    # pass 1: market breadth = share of universe stocks above their session VWAP
    above = {}
    for t, s in targets.items():
        if t in set(idx.values()):
            continue
        m5 = _load(engine, t, "5minute", since)
        if not m5.empty:
            above[s] = build.above_vwap(m5)
    breadth_df = pd.DataFrame(above)
    breadth = breadth_df.mean(axis=1).where(breadth_df.notna().sum(axis=1) >= max(1, len(above) // 2))

    events = exclusions.load_corporate_actions()
    for sym, days in _quality_jumps(engine).items():
        events.setdefault(sym, set()).update(days)

    stats = {"stocks": 0, "rows": 0, "labelled": 0, "excluded_rows": 0, "skipped": [], "per_symbol": {}}
    for t, sym in sorted(stocks.items(), key=lambda kv: kv[1]):
        m5 = _load(engine, t, "5minute", since)
        if len(m5) < 500:
            stats["skipped"].append(sym)
            continue
        f = build.stock_features(m5, _load(engine, t, "15minute", since), _load(engine, t, "day", daily_since))
        f = build.add_context(f, nifty_ctx, sector_ctx.get(sector_of.get(sym)), vix_ctx, breadth)
        lab = labels.triple_barrier(f, _exit_times(f, clock), **{k: LABEL_PARAMS[k]
                                                               for k in ("target_mult", "stop_mult")})
        f = pd.concat([f, lab], axis=1)

        sessions = sorted(f["day"].unique())
        reasons = exclusions.excluded_days(sessions, events.get(sym, set()), special)
        f["exclude_reason"] = f["day"].map(reasons)
        f["exclude"] = f["exclude_reason"].notna()
        member_days = [(vf, vt) for s, _, vf, vt in spells if s == sym]
        f["in_universe"] = f["day"].map(lambda d, md=member_days: any(vf <= d and (vt is None or d <= vt)
                                                                      for vf, vt in md))
        f["symbol"], f["instrument_token"], f["sector_index"] = sym, t, sector_of.get(sym)

        cols = (["ts", "decision_time", "day", "symbol", "instrument_token", "sector_index",
                 "open", "high", "low", "close", "volume", "atr14"]
                + build.ALL_FEATURES + labels.LABEL_COLUMNS + ["exclude", "exclude_reason", "in_universe"])
        out = f[cols].copy()
        out["day"] = pd.to_datetime(out["day"])
        buf = io.BytesIO()
        out.to_parquet(buf, index=False)
        put_bytes(f"features/{build.FEATURE_VERSION}/5minute/{sym}.parquet", buf.getvalue())

        n_lab = int(out["y_long"].notna().sum())
        stats["stocks"] += 1
        stats["rows"] += len(out)
        stats["labelled"] += n_lab
        stats["excluded_rows"] += int(out["exclude"].sum())
        stats["per_symbol"][sym] = {"rows": len(out), "labelled": n_lab,
                                    "first": str(out["day"].min().date()), "last": str(out["day"].max().date()),
                                    "long_hit_rate": round(float(out["y_long"].mean()), 4) if n_lab else None}

    manifest = {
        "version": build.FEATURE_VERSION, "built_at": now.isoformat(), "code_hash": code_hash(),
        "base_interval": "5minute", "features": build.ALL_FEATURES, "labels": labels.LABEL_COLUMNS,
        "label_params": LABEL_PARAMS, "exclusions": {"after_event_sessions": exclusions.AFTER_EVENT_SESSIONS,
                                                     "warmup_sessions": exclusions.WARMUP_SESSIONS},
        "special_sessions": sorted(str(d) for d in special), "stats": stats,
    }
    put_bytes(f"features/{build.FEATURE_VERSION}/manifest.json", json.dumps(manifest, indent=2).encode())
    summary = {k: v for k, v in stats.items() if k != "per_symbol"}
    rates = [v["long_hit_rate"] for v in stats["per_symbol"].values() if v["long_hit_rate"] is not None]
    summary["median_long_hit_rate"] = round(float(pd.Series(rates).median()), 4) if rates else None
    return summary
