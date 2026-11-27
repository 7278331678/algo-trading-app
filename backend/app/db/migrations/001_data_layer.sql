-- Phase 1: data layer

-- Kite instrument master (equities and indices we care about)
CREATE TABLE IF NOT EXISTS instruments (
    instrument_token BIGINT PRIMARY KEY,
    exchange_token   BIGINT,
    tradingsymbol    TEXT NOT NULL,
    name             TEXT,
    exchange         TEXT NOT NULL,
    segment          TEXT NOT NULL,
    instrument_type  TEXT NOT NULL,
    tick_size        NUMERIC(12, 4),
    lot_size         INTEGER,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_instruments_symbol ON instruments (exchange, tradingsymbol);

-- Point-in-time index membership. valid_to IS NULL means "currently a member".
-- Backtests ask "who was in the index on date D" so stocks that later left are not lost.
CREATE TABLE IF NOT EXISTS universe_members (
    id               BIGSERIAL PRIMARY KEY,
    index_name       TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    company_name     TEXT,
    industry         TEXT,
    sector_index     TEXT,
    isin             TEXT,
    instrument_token BIGINT,
    valid_from       DATE NOT NULL,
    valid_to         DATE,
    source           TEXT NOT NULL DEFAULT 'nse_csv',
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_universe_open
    ON universe_members (index_name, symbol) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS ix_universe_dates ON universe_members (index_name, valid_from, valid_to);

-- OHLCV candles for every interval in one table
CREATE TABLE IF NOT EXISTS candles (
    instrument_token BIGINT      NOT NULL,
    interval         TEXT        NOT NULL,   -- minute | 5minute | 15minute | day
    ts               TIMESTAMPTZ NOT NULL,   -- candle open time
    open             DOUBLE PRECISION NOT NULL,
    high             DOUBLE PRECISION NOT NULL,
    low              DOUBLE PRECISION NOT NULL,
    close            DOUBLE PRECISION NOT NULL,
    volume           BIGINT      NOT NULL DEFAULT 0,
    PRIMARY KEY (instrument_token, interval, ts)
);
CREATE INDEX IF NOT EXISTS ix_candles_interval_ts ON candles (interval, ts);

-- Every job run (backfill, daily update, purge, quality check, universe sync)
CREATE TABLE IF NOT EXISTS job_runs (
    id           BIGSERIAL PRIMARY KEY,
    job          TEXT NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    status       TEXT NOT NULL DEFAULT 'running',  -- running | ok | partial | failed | skipped
    stats        JSONB NOT NULL DEFAULT '{}'::jsonb,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS ix_job_runs_job ON job_runs (job, started_at DESC);

-- Findings from data-quality checks
CREATE TABLE IF NOT EXISTS data_quality_issues (
    id               BIGSERIAL PRIMARY KEY,
    run_id           BIGINT REFERENCES job_runs (id) ON DELETE CASCADE,
    instrument_token BIGINT NOT NULL,
    interval         TEXT NOT NULL,
    day              DATE,
    severity         TEXT NOT NULL,   -- critical | warning
    issue            TEXT NOT NULL,
    detail           TEXT
);
CREATE INDEX IF NOT EXISTS ix_dq_run ON data_quality_issues (run_id);

-- Stocks blocked from trading on a given day because their data failed a critical check
CREATE TABLE IF NOT EXISTS trading_blocks (
    instrument_token BIGINT NOT NULL,
    day              DATE   NOT NULL,
    reason           TEXT   NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (instrument_token, day)
);
