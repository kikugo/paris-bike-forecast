FROM python:3.12-slim

RUN pip install --no-cache-dir "fastapi==0.162.*" "uvicorn==0.54.*" "sqlalchemy==2.0.*" "psycopg[binary]==3.3.*"
WORKDIR /app
COPY ingest/__init__.py ingest/schema.py ingest/
COPY api/ api/
USER nobody
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
