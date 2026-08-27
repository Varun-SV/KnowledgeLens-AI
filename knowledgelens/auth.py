from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import uuid
from dataclasses import dataclass

from .database import Database

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


@dataclass(frozen=True, slots=True)
class AuthSettings:
    mode: str
    oidc_issuer: str | None
    oidc_client_id: str | None
    oidc_scopes: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "AuthSettings":
        mode = os.getenv("KNOWLEDGELENS_AUTH_MODE", "disabled").strip().casefold()
        if mode not in {"disabled", "local", "oidc"}:
            raise ValueError("KNOWLEDGELENS_AUTH_MODE must be disabled, local, or oidc.")
        issuer = os.getenv("KNOWLEDGELENS_OIDC_ISSUER", "").strip() or None
        client_id = os.getenv("KNOWLEDGELENS_OIDC_CLIENT_ID", "").strip() or None
        scopes = tuple(item for item in os.getenv("KNOWLEDGELENS_OIDC_SCOPES", "openid profile email").split() if item)
        if mode == "oidc" and (not issuer or not client_id):
            raise ValueError("OIDC auth requires KNOWLEDGELENS_OIDC_ISSUER and KNOWLEDGELENS_OIDC_CLIENT_ID.")
        return cls(mode, issuer, client_id, scopes)


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    if len(password) < 12:
        raise ValueError("Administrator passwords must contain at least 12 characters.")
    salt = salt or secrets.token_bytes(_SALT_BYTES)
    if len(salt) != _SALT_BYTES:
        raise ValueError(f"Password salts must contain exactly {_SALT_BYTES} bytes.")
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN
    )
    return (
        f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}$"
        + base64.urlsafe_b64encode(salt).decode("ascii")
        + "$"
        + base64.urlsafe_b64encode(derived).decode("ascii")
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n_raw, r_raw, p_raw, salt_raw, digest_raw = encoded.split("$", 5)
        if algorithm != "scrypt" or (int(n_raw), int(r_raw), int(p_raw)) != (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P):
            return False
        salt = base64.urlsafe_b64decode(salt_raw.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_raw.encode("ascii"))
        if len(salt) != _SALT_BYTES or len(expected) != _SCRYPT_DKLEN:
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, expected)


def bootstrap_admin(database: Database, username: str, password: str) -> bool:
    clean_username = username.strip()
    if not clean_username or len(clean_username) > 120:
        raise ValueError("Administrator username must be between 1 and 120 characters.")

    with database.connect() as connection:
        with connection.transaction():
            # Serialize first-admin creation without making the admin role globally unique;
            # public deployments may intentionally add more administrators later.
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (0x4B4C41444D494E,))
            existing = connection.execute("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1").fetchone()
            if existing:
                return False
            encoded = hash_password(password)
            connection.execute(
                "INSERT INTO users(id, username, password_hash, role) VALUES (%s, %s, %s, 'admin')",
                (uuid.uuid4(), clean_username, encoded),
            )
    return True
