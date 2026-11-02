#!/bin/bash
# Runs once, on the first start of the postgres container.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<SQL
CREATE USER airflow WITH PASSWORD '${AIRFLOW_DB_PASSWORD}';
CREATE DATABASE airflow OWNER airflow;
CREATE USER mlflow WITH PASSWORD '${MLFLOW_DB_PASSWORD}';
CREATE DATABASE mlflow OWNER mlflow;
CREATE USER bikes WITH PASSWORD '${BIKES_DB_PASSWORD}';
CREATE DATABASE bikes OWNER bikes;
CREATE USER grafana_ro WITH PASSWORD '${GRAFANA_DB_PASSWORD}';
SQL

# Grafana only reads the bikes tables, including ones created later by the bikes user.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname bikes <<SQL
GRANT CONNECT ON DATABASE bikes TO grafana_ro;
GRANT USAGE ON SCHEMA public TO grafana_ro;
ALTER DEFAULT PRIVILEGES FOR USER bikes IN SCHEMA public GRANT SELECT ON TABLES TO grafana_ro;
SQL
