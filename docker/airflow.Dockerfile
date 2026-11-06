FROM apache/airflow:3.3.2-python3.12

# LightGBM needs the OpenMP runtime to load the champion model.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Runtime deps of the ingest and forecast packages. Airflow already brings pandas, SQLAlchemy,
# psycopg and httpx at versions that satisfy these, so pip only adds what is missing.
COPY docker/airflow-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
