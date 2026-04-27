#!/bin/bash
set -e

echo "Running Airflow DB migrations..."
airflow db migrate

echo "Creating admin user..."
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  && echo "Admin user created." \
  || echo "Admin user already exists, skipping."

echo "Starting $1..."
exec airflow "$@"