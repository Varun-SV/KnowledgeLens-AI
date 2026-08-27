import os

import pytest

from knowledgelens.auth import bootstrap_admin
from knowledgelens.database import Database, DatabaseSettings
from knowledgelens.provider_activation import active_profile_record, set_active_profile
from knowledgelens.provider_profiles import ProviderProfileRepository


pytestmark = pytest.mark.skipif(
    not os.getenv("KNOWLEDGELENS_TEST_DATABASE_URL"),
    reason="PostgreSQL integration URL not configured",
)


def _database() -> Database:
    return Database(DatabaseSettings(url=os.environ["KNOWLEDGELENS_TEST_DATABASE_URL"]))


def test_schema_initialization_is_idempotent_and_seeds_profiles():
    database = _database()
    database.initialize()
    database.initialize()

    repository = ProviderProfileRepository(database)
    repository.seed_builtins()
    profiles = repository.list()
    assert {"Ollama / local", "llama.cpp / local", "OpenAI"} <= {profile.name for profile in profiles}

    with database.connect() as connection:
        version = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()["version"]
        tables = {
            row["table_name"]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ).fetchall()
        }
    assert version == 1
    assert {
        "workspaces",
        "documents",
        "jobs",
        "checkpoints",
        "provider_profiles",
        "stored_secrets",
        "app_settings",
        "users",
        "oidc_config",
    } <= tables


def test_active_profile_persists_and_bootstrap_admin_is_one_time(monkeypatch):
    database = _database()
    database.initialize()
    repository = ProviderProfileRepository(database)
    repository.seed_builtins()

    with database.connect() as connection:
        with connection.transaction():
            connection.execute("DELETE FROM app_settings WHERE key = 'active_provider_profile_id'")
            connection.execute("DELETE FROM users")

    openai = next(profile for profile in repository.list() if profile.name == "OpenAI")
    set_active_profile(database, openai.id)
    active = active_profile_record(database)
    assert active is not None
    assert str(active["id"]) == openai.id
    assert active["default_model"] == "gpt-4o-mini"
    assert os.environ["KNOWLEDGELENS_CUSTOM_ENDPOINT"] == "https://api.openai.com"
    assert os.environ["KNOWLEDGELENS_CUSTOM_MODEL"] == "gpt-4o-mini"

    assert bootstrap_admin(database, "integration-admin", "correct horse battery staple") is True
    assert bootstrap_admin(database, "second-admin", "correct horse battery staple") is False
    with database.connect() as connection:
        count = connection.execute("SELECT COUNT(*) AS count FROM users WHERE role = 'admin'").fetchone()["count"]
    assert count == 1
