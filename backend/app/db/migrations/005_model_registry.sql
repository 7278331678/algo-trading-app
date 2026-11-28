-- Phase 4 (finish): model registry. A run is a candidate until the weekly retrain decides:
--   rejected  failed the gate (holdout verdict not good enough); never traded, kept for the record
--   shadow    passed the gate; scored every week on days AFTER it was trained, no orders
--   champion  survived shadow on fresh data and beat the previous champion; the only model Phase 6 may use
--   retired   replaced, or its forward results went bad
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS spec JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS trained_until DATE;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS status_reason TEXT;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS status_changed_at TIMESTAMPTZ;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS mlflow_run_id TEXT;
CREATE INDEX IF NOT EXISTS ix_model_runs_status ON model_runs (status);

-- forward results of shadow/champion models on days they were never trained on
CREATE TABLE IF NOT EXISTS model_evals (
    id            BIGSERIAL PRIMARY KEY,
    model_run_id  BIGINT NOT NULL REFERENCES model_runs(id),
    evaluated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    from_day      DATE NOT NULL,
    to_day        DATE NOT NULL,
    trading_days  INT NOT NULL,
    summary       JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_model_evals_run ON model_evals (model_run_id, evaluated_at DESC)
