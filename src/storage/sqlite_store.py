"""SQLite storage backend — used for local development."""
import sqlite3
from pathlib import Path
from src.utils.config import DATA_DIR
from src.utils.models import LLMResponse, BrandMention, CitationSource
from src.storage.schema import SQLITE_DDL

DB_PATH = DATA_DIR / "tracker.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript(SQLITE_DDL)


def save_responses(responses: list[LLMResponse]):
    rows = [
        (
            r.run_id, r.prompt_id, r.prompt_text, r.provider.value,
            r.model, r.response_text, r.latency_ms, r.error,
            r.created_at.isoformat(),
        )
        for r in responses
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO llm_responses
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )


def save_mentions(mentions: list[BrandMention]):
    rows = [
        (
            m.run_id, m.prompt_id, m.provider.value,
            m.brand, m.position, m.sentiment.value, m.snippet,
            m.created_at.isoformat(),
        )
        for m in mentions
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO brand_mentions
               (run_id,prompt_id,provider,brand,position,sentiment,snippet,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )


def save_citations(citations: list[CitationSource]):
    rows = [
        (
            c.run_id, c.prompt_id, c.provider.value,
            c.url, c.domain, c.domain_type,
            c.created_at.isoformat(),
        )
        for c in citations
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO citation_sources
               (run_id,prompt_id,provider,url,domain,domain_type,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            rows,
        )


def query_df(sql: str, params=None):
    """Return query result as a pandas DataFrame."""
    import pandas as pd
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)
