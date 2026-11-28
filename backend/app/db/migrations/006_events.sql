-- Phase 5: corporate events, stored with the exact time they became public (IST).
-- Features may only use an event at a decision time AFTER its public_at.
CREATE TABLE IF NOT EXISTS corporate_announcements (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,                -- nse
    ext_id      TEXT NOT NULL,                -- exchange id, or a hash of symbol+time+subject
    symbol      TEXT NOT NULL,
    public_at   TIMESTAMPTZ NOT NULL,         -- when the exchange disseminated it
    subject     TEXT NOT NULL,                -- NSE "desc" (its own category label)
    body        TEXT,                         -- NSE "attchmntText" summary, if any
    attachment  TEXT,                         -- PDF/XML link on nsearchives
    category    TEXT NOT NULL,                -- our category (app/events/classify.py)
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, ext_id)
);
CREATE INDEX IF NOT EXISTS ix_ann_symbol_time ON corporate_announcements (symbol, public_at);

-- board meetings: the results calendar
CREATE TABLE IF NOT EXISTS board_meetings (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    meeting_date  DATE NOT NULL,
    purpose       TEXT NOT NULL,
    is_results    BOOLEAN NOT NULL,
    known_at      TIMESTAMPTZ NOT NULL,       -- intimation time, or a deliberately late estimate if unknown
    known_at_estimated BOOLEAN NOT NULL DEFAULT false,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, symbol, meeting_date, purpose)
);
CREATE INDEX IF NOT EXISTS ix_bm_symbol_date ON board_meetings (symbol, meeting_date);

-- what has been fetched, so backfills resume and daily runs only ask for new days
CREATE TABLE IF NOT EXISTS event_fetch_log (
    source      TEXT NOT NULL,
    kind        TEXT NOT NULL,                -- announcements | board_meetings
    symbol      TEXT NOT NULL,
    from_day    DATE NOT NULL,
    to_day      DATE NOT NULL,
    rows        INT NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, kind, symbol, from_day, to_day)
)
