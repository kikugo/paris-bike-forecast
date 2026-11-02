FROM apache/airflow:3.3.2-python3.12

# Runtime deps of the ingest package. Airflow already brings pandas, SQLAlchemy, psycopg
# and httpx at versions that satisfy these, so pip only adds what is missing.
COPY docker/airflow-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
