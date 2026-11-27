-- Phase 4: model training studies (artifacts under models/<run_id>/ in object storage)
CREATE TABLE IF NOT EXISTS model_runs (
    id                 BIGSERIAL PRIMARY KEY,
    run_id             TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    barrier            TEXT NOT NULL,
    verdict            TEXT NOT NULL,          -- no_edge | weak_positive | edge_candidate
    chosen             JSONB,                  -- model + threshold picked on the selection quarters
    chosen_on_holdout  JSONB,                  -- how that choice did on the untouched holdout quarters
    summary            JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifacts          TEXT,
    status             TEXT NOT NULL DEFAULT 'candidate'   -- candidate | shadow | champion | retired
);
CREATE INDEX IF NOT EXISTS ix_model_runs_created ON model_runs (created_at DESC)
