"""Unified storage interface — routes to SQLite or BigQuery based on config."""
import os
from pathlib import Path
from src.utils.config import GCP_PROJECT_ID, BIGQUERY_DATASET, GOOGLE_APPLICATION_CREDENTIALS
from src.utils.models import LLMResponse, BrandMention, CitationSource


def _bigquery_ready() -> bool:
    """BigQuery requires project + dataset + a credentials file that actually exists."""
    if not (GCP_PROJECT_ID and BIGQUERY_DATASET):
        return False
    creds = GOOGLE_APPLICATION_CREDENTIALS
    if not creds or "path/to" in creds or not Path(creds).exists():
        return False
    return True


_USE_BIGQUERY = _bigquery_ready()


def _backend():
    if _USE_BIGQUERY:
        from src.storage import bigquery_store
        return bigquery_store
    from src.storage import sqlite_store
    return sqlite_store


def init():
    if _USE_BIGQUERY:
        from src.storage.bigquery_store import init_dataset
        init_dataset()
    else:
        from src.storage.sqlite_store import init_db
        init_db()


def run_exists(run_id: str) -> bool:
    """Return True if this run_id has already been saved."""
    from src.utils.config import GCP_PROJECT_ID, BIGQUERY_DATASET
    if _USE_BIGQUERY:
        tbl = f"`{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.llm_responses`"
        sql = f"SELECT COUNT(*) FROM {tbl} WHERE run_id = '{run_id}'"
    else:
        sql = "SELECT COUNT(*) FROM llm_responses WHERE run_id = ?"
    df = query_df(sql, params=(run_id,) if not _USE_BIGQUERY else None)
    return int(df.iloc[0, 0]) > 0


def save_responses(responses: list[LLMResponse]):
    _backend().save_responses(responses)


def save_mentions(mentions: list[BrandMention]):
    _backend().save_mentions(mentions)


def save_citations(citations: list[CitationSource]):
    _backend().save_citations(citations)


def query_df(sql: str, params=None):
    """For SQLite pass params; BigQuery ignores params (use f-string safely)."""
    b = _backend()
    if _USE_BIGQUERY:
        return b.query_df(sql)
    return b.query_df(sql, params)
