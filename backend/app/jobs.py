"""Batch jobs, run as Kubernetes Jobs/CronJobs:  python -m app.jobs <command>

  migrate         apply database migrations
  universe-sync   refresh the Nifty 100 list, Kite tokens and sectors (weekly)
  backfill        download history for every interval, resumable (one-time, or to repair)
  daily-update    fetch today's candles, purge old data, run quality checks (16:05 on trading days)
  purge           delete candles older than the retention windows
  quality-check   check stored candles and block stocks with bad data
  build-features  indicators, market context and labels for every stock -> Parquet (after daily-update)
  backtest        run strategies x stop/target settings on the features, save results
  train-models    walk-forward models, threshold chosen on early quarters, judged on later ones
  weekly-retrain  score shadow/champion models on unseen days, promote/retire, train a new candidate
  signal-check    plant a known edge in the real data and confirm the pipeline finds it (and ignores noise)

Exit codes: 0 ok/skipped, 1 failed, 2 Zerodha login needed, 3 another run is in progress.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta

from app.config import Settings, get_settings
from app.core.market_clock import IST, HolidayCalendarMissing, MarketClock
from app.db import JobRun, advisory_lock, migrate
from app.logging_setup import setup_logging
from app.marketdata import history, quality, store, universe

log = logging.getLogger("jobs")


class NoKiteSession(RuntimeError):
    pass


# ------------------------------------------------------------------ plumbing
def _engine(settings: Settings):
    from sqlalchemy import create_engine

    return create_engine(settings.db_url, pool_pre_ping=True)


def _kite(settings: Settings, clock: MarketClock):
    import redis
    from kiteconnect import KiteConnect

    from app.state.token_store import TokenStore

    session = TokenStore(redis.Redis.from_url(settings.redis_url, decode_responses=True), clock).load()
    if not session:
        raise NoKiteSession("No Zerodha session. Log in at /auth/kite/login, then run again.")
    kite = KiteConnect(api_key=settings.kite_api_key, timeout=30)
    kite.set_access_token(session.access_token)
    return kite


# ------------------------------------------------------------------ steps
def step_universe_sync(settings: Settings, engine, kite, today: date) -> dict:
    constituents = universe.parse_constituents_csv(universe.fetch_constituents_csv(settings.universe_csv_url))
    if len(constituents) < 90:
        raise RuntimeError(f"constituents list has only {len(constituents)} rows; refusing to update")
    tokens = universe.upsert_instruments(engine, kite.instruments("NSE"))
    fo_count = universe.mark_fo_eligible(engine, kite.instruments("NFO"))
    stats = universe.sync_universe(
        engine, settings.universe_index, constituents, tokens, today,
        baseline_from=today - timedelta(days=settings.retention_day_days),
    )
    stats["constituents"] = len(constituents)
    stats["fo_eligible_stocks"] = fo_count
    if stats["missing_tokens"]:
        log.warning("no Kite token for: %s", ", ".join(stats["missing_tokens"]))
    if stats.get("baseline_assumed"):
        log.warning("first sync: past membership assumed equal to today's list until "
                    "data/universe_history.csv is filled (survivorship bias)")
    return stats


def step_fetch(settings: Settings, engine, kite, intervals: list[str], now: datetime,
               symbols: set[str] | None = None, refresh: bool = False) -> dict:
    today = now.date()
    retention = settings.retention_days
    since = today - timedelta(days=max(retention[i] for i in intervals))
    targets = universe.tokens_to_fetch(engine, settings.universe_index, since)
    if symbols:
        targets = {t: s for t, s in targets.items() if s in symbols}
    if not targets:
        raise RuntimeError("nothing to fetch: run universe-sync first (or check --symbols)")

    limiter = history.RateLimiter(settings.kite_hist_rate_per_sec)
    stats = {"instruments": len(targets), "requests": 0, "candles": 0, "failures": [], "by_interval": {}}
    started = time.monotonic()
    total_units = len(targets) * len(intervals)
    done_units = 0
    for interval in intervals:
        window_start = today - timedelta(days=retention[interval])
        added_here = 0
        for token, sym in sorted(targets.items(), key=lambda kv: kv[1]):
            try:
                if refresh:
                    store.delete_token_interval(engine, token, interval)
                start = history.resume_start(store.last_ts(engine, token, interval), window_start, interval)
                for c_start, c_end in history.plan_chunks(start, now, interval):
                    raw = history.fetch_chunk(kite, token, c_start, c_end, interval, limiter)
                    stats["requests"] += 1
                    added_here += store.upsert_candles(engine, token, interval,
                                                       history.normalize(raw, interval, now))
            except history.TokenExpired:
                raise
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                log.error("fetch failed %s %s: %s", sym, interval, exc)
                stats["failures"].append(f"{sym}/{interval}: {exc.__class__.__name__}")
            done_units += 1
            if done_units % 25 == 0 or done_units == total_units:
                elapsed = time.monotonic() - started
                eta = elapsed / done_units * (total_units - done_units)
                log.info("progress %d/%d (%s), %d requests, eta %.0f min",
                         done_units, total_units, interval, stats["requests"], eta / 60)
        stats["by_interval"][interval] = added_here
        stats["candles"] += added_here
    return stats


def step_purge(settings: Settings, engine) -> dict:
    return store.purge_older_than(engine, settings.retention_days)


def step_quality(settings: Settings, engine, run_id: int, clock: MarketClock, now: datetime,
                 days: int | None) -> dict:
    from sqlalchemy import text

    today = now.date()
    retention = settings.retention_days
    targets = universe.tokens_to_fetch(engine, settings.universe_index,
                                       today - timedelta(days=retention["day"]))
    with engine.connect() as conn:
        nifty = conn.execute(text("SELECT instrument_token FROM instruments WHERE segment='INDICES'"
                                  " AND tradingsymbol=:s"), {"s": universe.BROAD_INDEX}).scalar()
        index_tokens = {int(r[0]) for r in conn.execute(
            text("SELECT instrument_token FROM instruments WHERE segment='INDICES'"))}
    if nifty is None:
        raise RuntimeError("NIFTY 50 index token unknown: run universe-sync first")

    oldest = today - timedelta(days=retention["day"])
    ref_rows = store.load_candles(engine, nifty, "day", datetime.combine(oldest, datetime.min.time(), tzinfo=IST))
    reference = {r["ts"].astimezone(IST).date() for r in ref_rows}
    if not reference:
        raise RuntimeError("no NIFTY 50 daily candles yet: run backfill first")
    last_session = max(reference)

    # candle start times NIFTY 50 itself had each day: a stock is expected to have the same
    # candles, up to its own continuous-trading end (earlier for CAS/F&O stocks since Aug 2026)
    ref_times: dict[str, dict[date, list]] = {}
    with engine.connect() as conn:
        for interval in history.INTERVALS:
            if interval == "day":
                continue
            per_day: dict[date, list] = {}
            for ts, in conn.execute(text("SELECT ts FROM candles WHERE instrument_token=:t AND interval=:i"
                                         " ORDER BY ts"), {"t": nifty, "i": interval}):
                local = ts.astimezone(IST)
                per_day.setdefault(local.date(), []).append(local.time())
            ref_times[interval] = per_day
    short_sessions = sorted(
        d for d, times in ref_times.get("5minute", {}).items()
        if len(times) < 0.75 * history.CANDLES_PER_SESSION["5minute"]
    )
    # days on which auction (F&O) stocks stop continuous trading before the normal close
    all_days = {d for per in ref_times.values() for d in per}
    auction_end = {d: clock.rules.continuous_end(d, cas=True) for d in all_days
                   if clock.rules.continuous_end(d, cas=True) < clock.rules.continuous_end(d, cas=False)}

    counts = {"critical": 0, "warning": 0}
    blocked: list[str] = []
    issue_rows = []
    for token, sym in targets.items():
        token_critical_recent = False
        for interval in history.INTERVALS:
            start_day = today - timedelta(days=retention[interval])
            if days is not None:
                start_day = max(start_day, today - timedelta(days=days))
            rows = store.load_candles(engine, token, interval,
                                      datetime.combine(start_day, datetime.min.time(), tzinfo=IST))
            ref = {d for d in reference if d >= start_day}
            is_index = token in index_tokens
            for iss in quality.run_checks(rows, interval, ref, is_index=is_index,
                                          reference_times=ref_times.get(interval), auction_end=auction_end):
                counts[iss.severity] += 1
                issue_rows.append({"r": run_id, "t": token, "i": interval, "d": iss.day,
                                   "sev": iss.severity, "iss": iss.issue, "det": iss.detail[:500]})
                if iss.severity == "critical" and (iss.day is None or iss.day == last_session):
                    token_critical_recent = True
        if token_critical_recent and token not in index_tokens:
            blocked.append(sym)

    try:
        block_day = clock.next_trading_day(last_session)
    except HolidayCalendarMissing:
        block_day = None
        log.warning("holiday calendar missing for next year: trading blocks not written")

    with engine.begin() as conn:
        if issue_rows:
            conn.execute(text("INSERT INTO data_quality_issues (run_id, instrument_token, interval, day,"
                              " severity, issue, detail) VALUES (:r, :t, :i, :d, :sev, :iss, :det)"), issue_rows)
        if block_day:
            by_symbol = {s: t for t, s in targets.items()}
            for sym in blocked:
                conn.execute(text("INSERT INTO trading_blocks (instrument_token, day, reason) VALUES (:t, :d, :r)"
                                  " ON CONFLICT DO NOTHING"),
                             {"t": by_symbol[sym], "d": block_day, "r": "data quality: critical issue"})
    return {**counts, "blocked": sorted(blocked), "block_day": block_day, "last_session": last_session,
            "short_sessions": short_sessions}


# ------------------------------------------------------------------ commands
def _run(job: str, fn) -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    engine = _engine(settings)
    migrate(engine)
    with advisory_lock(engine, f"job:{job}") as got:
        if not got:
            log.error("%s is already running; exiting", job)
            return 3
        run = JobRun(engine, job)
        try:
            status = fn(settings, engine, run) or "ok"
            run.finish(status)
            log.info("%s finished: %s %s", job, status, run.stats)
            return 0
        except (NoKiteSession, history.TokenExpired) as exc:
            run.finish("skipped" if job == "daily-update" else "failed", str(exc))
            log.error("%s: Zerodha login needed: %s", job, exc)
            return 0 if job == "daily-update" else 2
        except Exception as exc:  # noqa: BLE001
            run.finish("failed", f"{exc.__class__.__name__}: {exc}")
            log.exception("%s failed", job)
            return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.jobs", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    sub.add_parser("universe-sync")
    bf = sub.add_parser("backfill")
    bf.add_argument("--intervals", nargs="+", choices=history.INTERVALS, default=list(history.INTERVALS))
    bf.add_argument("--symbols", nargs="+", help="limit to these trading symbols")
    bf.add_argument("--refresh", action="store_true", help="delete and re-download the selected series")
    sub.add_parser("daily-update")
    sub.add_parser("purge")
    qc = sub.add_parser("quality-check")
    qc.add_argument("--days", type=int, default=None, help="only check the last N days")
    bfe = sub.add_parser("build-features")
    bfe.add_argument("--symbols", nargs="+", help="limit to these trading symbols")
    bt = sub.add_parser("backtest")
    bt.add_argument("--strategies", nargs="+", default=["rules", "random"])
    bt.add_argument("--barriers", nargs="+", default=None, help="names from BARRIER_GRID (default: all)")
    bt.add_argument("--start", type=date.fromisoformat, default=None)
    bt.add_argument("--end", type=date.fromisoformat, default=None)
    bt.add_argument("--symbols", nargs="+", default=None)
    bt.add_argument("--slippage-bps", type=float, default=5.0)
    def model_args(sp, default_barrier="daily_0.4x0.6"):
        sp.add_argument("--barrier", default=default_barrier,
                        choices=["daily_0.4x0.6", "daily_0.6x1.2", "atr5_1.0x1.5"])
        sp.add_argument("--decision-times", nargs="+", default=None,
                        help="HH:MM candle-close times to decide at, e.g. 09:45 (default: every candle)")
        sp.add_argument("--slippage-bps", type=float, default=3.0, help="per side, also used in the cost-aware target")
        sp.add_argument("--models", nargs="+", default=["hgb", "logit"], choices=["hgb", "logit"])
        sp.add_argument("--max-train-rows", type=int, default=600_000)
        sp.add_argument("--symbols", nargs="+", default=None)

    model_args(sub.add_parser("train-models"))
    model_args(sub.add_parser("weekly-retrain", help="review shadows/champion on unseen days, then train a candidate"),
               default_barrier="daily_0.6x1.2")
    sc = sub.add_parser("signal-check", help="plant a known edge in the real data and check the pipeline finds it")
    model_args(sc, default_barrier="daily_0.6x1.2")
    sc.add_argument("--strengths", nargs="+", type=float, default=None)
    args = p.parse_args(argv)
    clock = MarketClock()

    if args.cmd == "migrate":
        settings = get_settings()
        setup_logging(settings.log_level)
        log.info("applied: %s", migrate(_engine(settings)) or "nothing new")
        return 0

    def universe_sync(settings, engine, run):
        run.stats = step_universe_sync(settings, engine, _kite(settings, clock), clock.now().date())

    def backfill(settings, engine, run):
        kite = _kite(settings, clock)
        now = clock.now()
        if not universe.members_on(engine, settings.universe_index, now.date()):
            run.stats["universe"] = step_universe_sync(settings, engine, kite, now.date())
        run.stats["fetch"] = step_fetch(settings, engine, kite, args.intervals, now,
                                        set(args.symbols) if args.symbols else None, args.refresh)
        run.stats["quality"] = step_quality(settings, engine, run.id, clock, now, days=None)
        return "partial" if run.stats["fetch"]["failures"] else "ok"

    def daily_update(settings, engine, run):
        now = clock.now()
        try:
            if not clock.is_trading_day(now.date()):
                run.stats["reason"] = "not a trading day"
                return "skipped"
        except HolidayCalendarMissing as exc:
            log.warning("%s; fetching anyway", exc)
        kite = _kite(settings, clock)
        run.stats["fetch"] = step_fetch(settings, engine, kite, list(history.INTERVALS), now)
        run.stats["purge"] = step_purge(settings, engine)
        run.stats["quality"] = step_quality(settings, engine, run.id, clock, now, days=7)
        return "partial" if run.stats["fetch"]["failures"] else "ok"

    def purge(settings, engine, run):
        run.stats = step_purge(settings, engine)

    def quality_check(settings, engine, run):
        run.stats = step_quality(settings, engine, run.id, clock, clock.now(), days=args.days)

    def build_features(settings, engine, run):
        from app.features.pipeline import build_universe_features
        from app.providers.storage import ObjectStore

        # never build while today's candles are still being written
        for _ in range(60):
            with advisory_lock(engine, "job:daily-update") as free:
                if free:
                    break
            log.info("daily-update is running; waiting")
            time.sleep(30)
        store_ = ObjectStore(settings)
        store_.ensure_bucket()
        run.stats = build_universe_features(settings, engine, clock, store_.put_bytes,
                                            set(args.symbols) if args.symbols else None)
        return "partial" if run.stats["skipped"] else "ok"

    def backtest(settings, engine, run):
        from app.backtest import runner, strategies
        from app.backtest.engine import BacktestConfig
        from app.providers.storage import ObjectStore

        store_ = ObjectStore(settings)
        df = runner.load_features(store_.list_keys, store_.get_bytes, args.start, args.end,
                                  set(args.symbols) if args.symbols else None)
        df = runner.add_exit_times(df, clock)
        grid = [g for g in strategies.BARRIER_GRID if not args.barriers or g[0] in args.barriers]
        base = BacktestConfig(capital=settings.capital, risk_pct=settings.risk_per_trade_pct,
                              max_trades_per_day=settings.max_trades_per_day,
                              daily_loss_cap_pct=settings.daily_loss_cap_pct, slippage_bps=args.slippage_bps)
        results = runner.run_study(df, base, args.strategies, grid)
        run_id = clock.now().strftime("%Y%m%d-%H%M%S")
        period = {"start": str(df["day"].min()), "end": str(df["day"].max()), "symbols": int(df["symbol"].nunique())}
        runner.save_results(engine, store_.put_bytes, run_id, results, period)
        run.stats = {"run_id": run_id, **period,
                     "results": {r["name"]: {k: r["summary"].get(k) for k in
                                             ("trades", "net_pnl", "gross_before_costs", "cost_per_trade",
                                              "profit_factor", "win_rate", "max_drawdown_pct",
                                              "profitable_quarters", "would_halt_on")} for r in results}}

    def model_setup(settings):
        from datetime import time as dtime

        from app.backtest.engine import BacktestConfig
        from app.models import dataset
        from app.providers.storage import ObjectStore

        store_ = ObjectStore(settings)
        df = dataset.load(store_.list_keys, store_.get_bytes, set(args.symbols) if args.symbols else None)
        base = BacktestConfig(capital=settings.capital, risk_pct=settings.risk_per_trade_pct,
                              max_trades_per_day=settings.max_trades_per_day,
                              daily_loss_cap_pct=settings.daily_loss_cap_pct)
        cost = dataset.CostModel(slippage_bps_per_side=args.slippage_bps)
        dts = [dtime.fromisoformat(t) for t in args.decision_times] if args.decision_times else None
        spec = {"barrier": args.barrier, "decision_times": [t.strftime("%H:%M") for t in dts] if dts else None,
                "slippage_bps": args.slippage_bps, "kinds": list(args.models), "max_train_rows": args.max_train_rows,
                "symbols": args.symbols}
        return store_, df, base, cost, dts, spec

    def train_and_save(settings, engine, store_, df, base, cost, dts, spec) -> dict:
        from app.models import study, tracking, walkforward

        kinds = tuple(spec["kinds"])
        result = study.run_study(df, clock, base, spec["barrier"], cost, kinds,
                                 max_train_rows=spec["max_train_rows"], decision_times=dts)
        final = walkforward.fit_final(result["_df"], result["_features"], result["_final_mask"], kinds,
                                      spec["max_train_rows"])
        run_id = clock.now().strftime("%Y%m%d-%H%M%S")
        pk = study.save(engine, store_.put_bytes, run_id, result, final, kinds, spec)
        mlflow_id = tracking.log_training_run(settings.mlflow_tracking_uri, run_id, spec, result,
                                              {"model_run_id": pk})
        if mlflow_id:
            from sqlalchemy import text

            with engine.begin() as conn:
                conn.execute(text("UPDATE model_runs SET mlflow_run_id = :m WHERE id = :i"), {"m": mlflow_id, "i": pk})
        return {"model_run_id": pk, "run_id": run_id, "mlflow_run_id": mlflow_id, "verdict": result["verdict"],
                "trained_until": str(study.trained_until(result)),
                "base_win_rate_after_costs": result["base_win_rate_after_costs"],
                "chosen": result["chosen"], "chosen_on_holdout": result["chosen_on_holdout"],
                "auc_ensemble": {m["quarter"] + "/" + m["side"]: m["auc"] for m in result["fold_metrics"]
                                 if m["model"] == "ensemble"}}

    def train_models(settings, engine, run):
        store_, df, base, cost, dts, spec = model_setup(settings)
        run.stats = train_and_save(settings, engine, store_, df, base, cost, dts, spec)

    def weekly_retrain(settings, engine, run):
        from app.models import registry

        store_, df, base, cost, dts, spec = model_setup(settings)
        run.stats["review"] = registry.weekly_review(engine, store_.get_bytes, df, clock, base)
        run.stats["candidate"] = train_and_save(settings, engine, store_, df, base, cost, dts, spec)
        status, why = registry.register_candidate(engine, run.stats["candidate"]["model_run_id"],
                                                  run.stats["candidate"]["verdict"])
        run.stats["candidate"]["status"] = status
        run.stats["candidate"]["status_reason"] = why

    def signal_check(settings, engine, run):
        from app.models import signal_check as sc

        _, df, base, cost, dts, spec = model_setup(settings)
        res = sc.run(df, clock, base, spec["barrier"], cost, tuple(spec["kinds"]), dts,
                     strengths=tuple(args.strengths) if args.strengths else sc.DEFAULT_STRENGTHS,
                     max_train_rows=spec["max_train_rows"])
        run.stats = {"spec": spec, **res}
        return "ok" if res["passed"] else "failed_check"

    handlers = {"universe-sync": universe_sync, "backfill": backfill, "daily-update": daily_update,
                "purge": purge, "quality-check": quality_check, "build-features": build_features,
                "backtest": backtest, "train-models": train_models, "weekly-retrain": weekly_retrain,
                "signal-check": signal_check}
    return _run(args.cmd, handlers[args.cmd])


if __name__ == "__main__":
    sys.exit(main())
