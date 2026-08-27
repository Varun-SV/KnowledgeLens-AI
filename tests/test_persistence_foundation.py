from io import BytesIO

import pytest

from knowledgelens.auth import AuthSettings, hash_password, verify_password
from knowledgelens.blob_store import LocalContentStore
from knowledgelens.database import Database, DatabaseSettings, schema_statements
from knowledgelens.provider_profiles import legacy_profiles, profile_management_error
from knowledgelens.secrets import CompositeSecretStore, MemorySecretStore, _fernet_from_master_key


class FailingStore:
    def get(self, secret_ref):
        raise RuntimeError("unavailable")

    def set(self, secret_ref, value):
        raise RuntimeError("unavailable")

    def delete(self, secret_ref):
        raise RuntimeError("unavailable")


def test_database_is_optional_and_schema_contains_v03_foundation(monkeypatch):
    monkeypatch.delenv("KNOWLEDGELENS_DATABASE_URL", raising=False)
    settings = DatabaseSettings.from_env()
    assert Database(settings).enabled is False
    sql = "\n".join(schema_statements())
    for table in ("workspaces", "documents", "jobs", "checkpoints", "provider_profiles", "stored_secrets", "users", "oidc_config"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql


def test_blob_store_streams_and_deduplicates(tmp_path):
    store = LocalContentStore(tmp_path / "blobs")
    first = store.put_stream(BytesIO(b"abc" * 1000), chunk_size=17)
    second = store.put_bytes(b"abc" * 1000)
    assert first.sha256 == second.sha256
    assert first.byte_size == 3000
    assert store.exists(first.ref)
    with store.open(first.ref) as handle:
        assert handle.read() == b"abc" * 1000


def test_blob_store_rejects_invalid_reference(tmp_path):
    store = LocalContentStore(tmp_path)
    with pytest.raises(ValueError):
        store.exists("../../etc/passwd")


def test_secret_fallback_and_fernet_derivation():
    fallback = MemorySecretStore({})
    store = CompositeSecretStore(FailingStore(), fallback)
    store.set("provider:x", "secret")
    assert store.get("provider:x") == "secret"
    with pytest.raises(ValueError):
        _fernet_from_master_key("short")
    fernet = _fernet_from_master_key("a" * 32)
    assert fernet.decrypt(fernet.encrypt(b"secret")) == b"secret"


def test_provider_policy_preserves_public_admin_boundary():
    assert profile_management_error(deployment_mode="local", role=None) is None
    assert profile_management_error(deployment_mode="private", role=None) is None
    assert profile_management_error(deployment_mode="public", role=None)
    assert profile_management_error(deployment_mode="public", role="editor")
    assert profile_management_error(deployment_mode="public", role="admin") is None
    assert [profile.name for profile in legacy_profiles()] == ["Ollama / local", "llama.cpp / local", "OpenAI"]


def test_password_hash_and_oidc_validation(monkeypatch):
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("incorrect horse battery staple", encoded)
    monkeypatch.setenv("KNOWLEDGELENS_AUTH_MODE", "oidc")
    monkeypatch.delenv("KNOWLEDGELENS_OIDC_ISSUER", raising=False)
    monkeypatch.delenv("KNOWLEDGELENS_OIDC_CLIENT_ID", raising=False)
    with pytest.raises(ValueError):
        AuthSettings.from_env()
