from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - dependency/install failure path
    psycopg = None
    dict_row = None


class DatabaseUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    url: str | None
    connect_timeout_seconds: int = 5

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        url = os.getenv("KNOWLEDGELENS_DATABASE_URL", "").strip() or None
        raw_timeout = os.getenv("KNOWLEDGELENS_DATABASE_CONNECT_TIMEOUT", "5").strip()
        try:
            timeout = max(1, min(int(raw_timeout), 30))
        except ValueError:
            timeout = 5
        return cls(url=url, connect_timeout_seconds=timeout)


SCHEMA_VERSION = 1

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workspaces (
        id UUID PRIMARY KEY,
        name TEXT NOT NULL,
        mode TEXT NOT NULL DEFAULT 'standard',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY,
        workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        media_type TEXT,
        byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
        content_sha256 CHAR(64) NOT NULL,
        blob_ref TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'imported',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id)",
    "CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(content_sha256)",
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id UUID PRIMARY KEY,
        workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
        document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        progress_current BIGINT NOT NULL DEFAULT 0 CHECK (progress_current >= 0),
        progress_total BIGINT CHECK (progress_total IS NULL OR progress_total >= 0),
        state_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_workspace_status ON jobs(workspace_id, status)",
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        id UUID PRIMARY KEY,
        job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        sequence BIGINT NOT NULL CHECK (sequence >= 0),
        cursor_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(job_id, sequence)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_profiles (
        id UUID PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        provider_type TEXT NOT NULL,
        base_url TEXT NOT NULL,
        default_model TEXT NOT NULL,
        capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
        secret_ref TEXT,
        is_builtin BOOLEAN NOT NULL DEFAULT FALSE,
        enabled BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stored_secrets (
        secret_ref TEXT PRIMARY KEY,
        ciphertext BYTEA NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value_json JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY,
        username TEXT UNIQUE,
        password_hash TEXT,
        oidc_subject TEXT UNIQUE,
        role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
        disabled BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CHECK (username IS NOT NULL OR oidc_subject IS NOT NULL)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS oidc_config (
        id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
        issuer TEXT,
        client_id TEXT,
        scopes TEXT[] NOT NULL DEFAULT ARRAY['openid', 'profile', 'email'],
        enabled BOOLEAN NOT NULL DEFAULT FALSE,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
)


class Database:
    def __init__(self, settings: DatabaseSettings | None = None):
        self.settings = settings or DatabaseSettings.from_env()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.url)

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if not self.settings.url:
            raise DatabaseUnavailable(
                "PostgreSQL is not configured. Set KNOWLEDGELENS_DATABASE_URL to enable persistent workspaces."
            )
        if psycopg is None:
            raise DatabaseUnavailable('PostgreSQL support requires the "psycopg[binary]" dependency.')
        try:
            connection = psycopg.connect(
                self.settings.url,
                connect_timeout=self.settings.connect_timeout_seconds,
                row_factory=dict_row,
            )
        except Exception as exc:
            raise DatabaseUnavailable("Could not connect to the configured PostgreSQL database.") from exc
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        if not self.enabled:
            return
        with self.connect() as connection:
            with connection.transaction():
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version) VALUES (%s) ON CONFLICT (version) DO NOTHING",
                    (SCHEMA_VERSION,),
                )

    def healthcheck(self) -> bool:
        if not self.enabled:
            return False
        with self.connect() as connection:
            row = connection.execute("SELECT 1 AS ok").fetchone()
            return bool(row and row["ok"] == 1)


def schema_statements() -> tuple[str, ...]:
    """Expose immutable migration SQL for tests/diagnostics."""
    return _SCHEMA_STATEMENTS
