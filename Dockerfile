FROM python:3.11-slim

WORKDIR /app

COPY environment.yml .
RUN pip install --no-cache-dir \
    openai anthropic google-genai \
    fastapi uvicorn pydantic python-dotenv \
    pandas google-cloud-bigquery db-dtypes \
    lightgbm scikit-learn plotly dash \
    spacy tenacity rich && \
    python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY .env.example .env

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
