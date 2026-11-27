-- Phase 3: backtest results (artifacts live in object storage under backtests/<run_id>/)
CREATE TABLE IF NOT EXISTS backtest_runs (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    name        TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    barrier     TEXT NOT NULL,
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary     JSONB NOT NULL DEFAULT '{}'::jsonb,
    periods     JSONB NOT NULL DEFAULT '[]'::jsonb,
    artifacts   TEXT
);
CREATE INDEX IF NOT EXISTS ix_backtest_runs_run ON backtest_runs (run_id);
CREATE INDEX IF NOT EXISTS ix_backtest_runs_created ON backtest_runs (created_at DESC)
