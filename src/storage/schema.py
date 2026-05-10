"""Shared table/schema definitions for both SQLite and BigQuery."""

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS llm_responses (
    run_id       TEXT NOT NULL,
    prompt_id    TEXT NOT NULL,
    prompt_text  TEXT,
    provider     TEXT NOT NULL,
    model        TEXT,
    response_text TEXT,
    latency_ms   INTEGER,
    error        TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, prompt_id, provider)
);

CREATE TABLE IF NOT EXISTS brand_mentions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    prompt_id    TEXT NOT NULL,
    provider     TEXT NOT NULL,
    brand        TEXT NOT NULL,
    position     INTEGER,
    sentiment    TEXT,
    snippet      TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS citation_sources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    prompt_id    TEXT NOT NULL,
    provider     TEXT NOT NULL,
    url          TEXT,
    domain       TEXT,
    domain_type  TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_brand_mentions_brand ON brand_mentions(brand);
CREATE INDEX IF NOT EXISTS idx_brand_mentions_run ON brand_mentions(run_id);
CREATE INDEX IF NOT EXISTS idx_brand_mentions_date ON brand_mentions(created_at);
"""

BIGQUERY_SCHEMAS = {
    "llm_responses": [
        ("run_id", "STRING"),
        ("prompt_id", "STRING"),
        ("prompt_text", "STRING"),
        ("provider", "STRING"),
        ("model", "STRING"),
        ("response_text", "STRING"),
        ("latency_ms", "INTEGER"),
        ("error", "STRING"),
        ("created_at", "TIMESTAMP"),
    ],
    "brand_mentions": [
        ("run_id", "STRING"),
        ("prompt_id", "STRING"),
        ("provider", "STRING"),
        ("brand", "STRING"),
        ("position", "INTEGER"),
        ("sentiment", "STRING"),
        ("snippet", "STRING"),
        ("created_at", "TIMESTAMP"),
    ],
    "citation_sources": [
        ("run_id", "STRING"),
        ("prompt_id", "STRING"),
        ("provider", "STRING"),
        ("url", "STRING"),
        ("domain", "STRING"),
        ("domain_type", "STRING"),
        ("created_at", "TIMESTAMP"),
    ],
}
