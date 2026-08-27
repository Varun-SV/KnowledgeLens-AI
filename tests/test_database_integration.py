import os

import pytest

from knowledgelens.database import Database, DatabaseSettings
from knowledgelens.provider_profiles import ProviderProfileRepository


@pytest.mark.skipif(not os.getenv("KNOWLEDGELENS_TEST_DATABASE_URL"), reason="PostgreSQL integration URL not configured")
def test_schema_initialization_is_idempotent_and_seeds_profiles():
    database = Database(DatabaseSettings(url=os.environ["KNOWLEDGELENS_TEST_DATABASE_URL"]))
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
    assert {"workspaces", "documents", "jobs", "checkpoints", "provider_profiles", "stored_secrets"} <= tables
