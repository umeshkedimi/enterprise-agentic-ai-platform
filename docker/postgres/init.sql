-- Runs against the default database (POSTGRES_DB) on first container start.
CREATE EXTENSION IF NOT EXISTS vector;

-- Separate database for integration tests, isolated from local dev data.
SELECT 'CREATE DATABASE ekap_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'ekap_test')\gexec

\connect ekap_test
CREATE EXTENSION IF NOT EXISTS vector;
