"""BigQuery storage backend — used for production."""
from google.cloud import bigquery
from src.utils.config import GCP_PROJECT_ID, BIGQUERY_DATASET
from src.utils.models import LLMResponse, BrandMention, CitationSource
from src.storage.schema import BIGQUERY_SCHEMAS

_client: bigquery.Client | None = None


def get_client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=GCP_PROJECT_ID)
    return _client


def _table_ref(table: str) -> str:
    return f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{table}"


def init_dataset():
    """Create dataset and tables if they don't exist."""
    client = get_client()
    dataset_ref = bigquery.Dataset(f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}")
    dataset_ref.location = "US"
    client.create_dataset(dataset_ref, exists_ok=True)

    for table_name, columns in BIGQUERY_SCHEMAS.items():
        schema = [
            bigquery.SchemaField(name, dtype)
            for name, dtype in columns
        ]
        table = bigquery.Table(_table_ref(table_name), schema=schema)
        client.create_table(table, exists_ok=True)


def save_responses(responses: list[LLMResponse]):
    rows = [
        {
            "run_id": r.run_id, "prompt_id": r.prompt_id,
            "prompt_text": r.prompt_text, "provider": r.provider.value,
            "model": r.model, "response_text": r.response_text,
            "latency_ms": r.latency_ms, "error": r.error,
            "created_at": r.created_at.isoformat(),
        }
        for r in responses
    ]
    if not rows:
        return
    errors = get_client().insert_rows_json(_table_ref("llm_responses"), rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")


def save_mentions(mentions: list[BrandMention]):
    rows = [
        {
            "run_id": m.run_id, "prompt_id": m.prompt_id,
            "provider": m.provider.value, "brand": m.brand,
            "position": m.position, "sentiment": m.sentiment.value,
            "snippet": m.snippet, "created_at": m.created_at.isoformat(),
        }
        for m in mentions
    ]
    if not rows:
        return
    errors = get_client().insert_rows_json(_table_ref("brand_mentions"), rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")


def save_citations(citations: list[CitationSource]):
    rows = [
        {
            "run_id": c.run_id, "prompt_id": c.prompt_id,
            "provider": c.provider.value, "url": c.url,
            "domain": c.domain, "domain_type": c.domain_type,
            "created_at": c.created_at.isoformat(),
        }
        for c in citations
    ]
    if not rows:
        return
    errors = get_client().insert_rows_json(_table_ref("citation_sources"), rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")


def query_df(sql: str):
    """Return BigQuery query result as a pandas DataFrame."""
    return get_client().query(sql).to_dataframe()
