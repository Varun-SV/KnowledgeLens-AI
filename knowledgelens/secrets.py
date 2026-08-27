from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from typing import Protocol

from cryptography.fernet import Fernet, InvalidToken

from .database import Database, DatabaseUnavailable

try:
    import keyring
    from keyring.errors import KeyringError
except ImportError:  # pragma: no cover - dependency/install failure path
    keyring = None

    class KeyringError(Exception):
        pass


SERVICE_NAME = "KnowledgeLens AI"


class SecretStore(Protocol):
    def get(self, secret_ref: str) -> str | None: ...
    def set(self, secret_ref: str, value: str) -> None: ...
    def delete(self, secret_ref: str) -> None: ...


@dataclass(slots=True)
class MemorySecretStore:
    values: dict[str, str]

    def get(self, secret_ref: str) -> str | None:
        return self.values.get(secret_ref)

    def set(self, secret_ref: str, value: str) -> None:
        self.values[secret_ref] = value

    def delete(self, secret_ref: str) -> None:
        self.values.pop(secret_ref, None)


class KeyringSecretStore:
    def __init__(self, service_name: str = SERVICE_NAME):
        self.service_name = service_name

    def _require_backend(self) -> None:
        if keyring is None:
            raise RuntimeError("OS keyring support is unavailable.")

    def get(self, secret_ref: str) -> str | None:
        self._require_backend()
        try:
            return keyring.get_password(self.service_name, secret_ref)
        except KeyringError as exc:
            raise RuntimeError("The OS credential store is unavailable.") from exc

    def set(self, secret_ref: str, value: str) -> None:
        self._require_backend()
        try:
            keyring.set_password(self.service_name, secret_ref, value)
        except KeyringError as exc:
            raise RuntimeError("The OS credential store is unavailable.") from exc

    def delete(self, secret_ref: str) -> None:
        self._require_backend()
        try:
            keyring.delete_password(self.service_name, secret_ref)
        except KeyringError:
            return


def _fernet_from_master_key(master_key: str) -> Fernet:
    if len(master_key) < 32:
        raise ValueError("KNOWLEDGELENS_MASTER_KEY must be at least 32 characters.")
    digest = hashlib.sha256(master_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


class EncryptedDatabaseSecretStore:
    def __init__(self, database: Database, master_key: str | None = None):
        self.database = database
        configured = master_key if master_key is not None else os.getenv("KNOWLEDGELENS_MASTER_KEY", "")
        self.fernet = _fernet_from_master_key(configured)

    def get(self, secret_ref: str) -> str | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT ciphertext FROM stored_secrets WHERE secret_ref = %s",
                (secret_ref,),
            ).fetchone()
        if not row:
            return None
        try:
            return self.fernet.decrypt(bytes(row["ciphertext"])).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError) as exc:
            raise RuntimeError("Stored secret could not be decrypted with the configured master key.") from exc

    def set(self, secret_ref: str, value: str) -> None:
        ciphertext = self.fernet.encrypt(value.encode("utf-8"))
        with self.database.connect() as connection:
            with connection.transaction():
                connection.execute(
                    """
                    INSERT INTO stored_secrets(secret_ref, ciphertext)
                    VALUES (%s, %s)
                    ON CONFLICT (secret_ref) DO UPDATE
                    SET ciphertext = EXCLUDED.ciphertext, updated_at = NOW()
                    """,
                    (secret_ref, ciphertext),
                )

    def delete(self, secret_ref: str) -> None:
        with self.database.connect() as connection:
            with connection.transaction():
                connection.execute("DELETE FROM stored_secrets WHERE secret_ref = %s", (secret_ref,))


class CompositeSecretStore:
    """Prefer OS keyring; fall back to encrypted PostgreSQL when configured."""

    def __init__(self, primary: SecretStore, fallback: SecretStore | None = None):
        self.primary = primary
        self.fallback = fallback

    def get(self, secret_ref: str) -> str | None:
        try:
            value = self.primary.get(secret_ref)
        except RuntimeError:
            value = None
        if value is not None or self.fallback is None:
            return value
        return self.fallback.get(secret_ref)

    def set(self, secret_ref: str, value: str) -> None:
        try:
            self.primary.set(secret_ref, value)
            return
        except RuntimeError:
            if self.fallback is None:
                raise
        self.fallback.set(secret_ref, value)

    def delete(self, secret_ref: str) -> None:
        try:
            self.primary.delete(secret_ref)
        except RuntimeError:
            pass
        if self.fallback is not None:
            self.fallback.delete(secret_ref)


def build_secret_store(database: Database) -> SecretStore:
    primary = KeyringSecretStore()
    master_key = os.getenv("KNOWLEDGELENS_MASTER_KEY", "")
    fallback: SecretStore | None = None
    if database.enabled and master_key:
        try:
            fallback = EncryptedDatabaseSecretStore(database, master_key)
        except (ValueError, DatabaseUnavailable):
            fallback = None
    return CompositeSecretStore(primary, fallback)
