-- Phase 5: every poll of an NSE RSS feed, so coverage gaps are measured, not assumed
CREATE TABLE IF NOT EXISTS rss_polls (
    id        BIGSERIAL PRIMARY KEY,
    feed      TEXT NOT NULL,
    polled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    items     INT NOT NULL,
    kept      INT NOT NULL,
    unparsed  INT NOT NULL,
    oldest    TIMESTAMPTZ,
    newest    TIMESTAMPTZ,
    gap       BOOLEAN NOT NULL DEFAULT false   -- oldest item newer than the previous newest: filings may be missing
);
CREATE INDEX IF NOT EXISTS ix_rss_polls_feed_time ON rss_polls (feed, polled_at DESC)
