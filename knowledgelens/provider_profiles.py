from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Iterable

from .database import Database
from .http_client import get_pinned
from .limits import MAX_MODEL_NAME_CHARS
from .security import resolve_endpoint

KNOWN_CAPABILITIES = ("text", "vision", "structured_output")
KNOWN_PROVIDER_TYPES = ("openai-compatible", "openai", "ollama", "llama.cpp")
MAX_DISCOVERED_MODELS = 1_000


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    id: str
    name: str
    provider_type: str
    base_url: str
    default_model: str
    capabilities: tuple[str, ...] = ("text",)
    secret_ref: str | None = None
    is_builtin: bool = False
    enabled: bool = True

    @property
    def requires_api_key(self) -> bool:
        return self.provider_type == "openai"


LEGACY_PROFILES = (
    ProviderProfile(
        "legacy-ollama",
        "Ollama / local",
        "ollama",
        "http://localhost:11434",
        "llama3.1",
        ("text",),
        is_builtin=True,
    ),
    ProviderProfile(
        "legacy-llamacpp",
        "llama.cpp / local",
        "llama.cpp",
        "http://localhost:8080",
        "llama3.1",
        ("text",),
        is_builtin=True,
    ),
    ProviderProfile(
        "legacy-openai",
        "OpenAI",
        "openai",
        "https://api.openai.com",
        "gpt-4o-mini",
        ("text", "vision", "structured_output"),
        is_builtin=True,
    ),
)


class ProviderProfileRepository:
    def __init__(self, database: Database):
        self.database = database

    def seed_builtins(self) -> None:
        if not self.database.enabled:
            return
        for profile in LEGACY_PROFILES:
            profile_id = uuid.uuid5(uuid.NAMESPACE_URL, f"knowledgelens:provider:{profile.name}")
            with self.database.connect() as connection:
                with connection.transaction():
                    connection.execute(
                        """
                        INSERT INTO provider_profiles(
                            id, name, provider_type, base_url, default_model, capabilities, is_builtin, enabled
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, TRUE, TRUE)
                        ON CONFLICT (name) DO NOTHING
                        """,
                        (
                            profile_id,
                            profile.name,
                            profile.provider_type,
                            profile.base_url,
                            profile.default_model,
                            json.dumps(profile.capabilities),
                        ),
                    )

    def list(self) -> list[ProviderProfile]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, provider_type, base_url, default_model, capabilities, secret_ref, is_builtin, enabled
                FROM provider_profiles WHERE enabled = TRUE ORDER BY is_builtin DESC, name ASC
                """
            ).fetchall()
        return [_profile_from_row(row) for row in rows]

    def get(self, profile_id: str) -> ProviderProfile | None:
        try:
            profile_uuid = uuid.UUID(profile_id)
        except (ValueError, AttributeError, TypeError):
            return None
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, provider_type, base_url, default_model, capabilities, secret_ref, is_builtin, enabled
                FROM provider_profiles WHERE id = %s
                """,
                (profile_uuid,),
            ).fetchone()
        return _profile_from_row(row) if row else None

    def save(
        self,
        *,
        name: str,
        provider_type: str,
        base_url: str,
        default_model: str,
        capabilities: Iterable[str],
        secret_ref: str | None = None,
        profile_id: str | None = None,
        is_builtin: bool = False,
    ) -> ProviderProfile:
        clean_name = " ".join(name.split())
        clean_type = provider_type.strip().casefold()
        clean_url = base_url.strip().rstrip("/")
        clean_model = " ".join(default_model.split())
        if not clean_name or len(clean_name) > 120:
            raise ValueError("Provider profile name must be between 1 and 120 characters.")
        if clean_type not in KNOWN_PROVIDER_TYPES:
            raise ValueError("Unsupported provider profile type.")
        if not clean_model or len(clean_model) > MAX_MODEL_NAME_CHARS:
            raise ValueError(f"Default model must be between 1 and {MAX_MODEL_NAME_CHARS} characters.")
        resolve_endpoint(clean_url)
        normalized_caps = tuple(sorted({item for item in capabilities if item in KNOWN_CAPABILITIES} or {"text"}))
        profile_uuid = uuid.UUID(profile_id) if profile_id else uuid.uuid4()

        with self.database.connect() as connection:
            with connection.transaction():
                existing = connection.execute(
                    "SELECT is_builtin FROM provider_profiles WHERE id = %s",
                    (profile_uuid,),
                ).fetchone()
                if is_builtin or (existing and bool(existing["is_builtin"])):
                    raise ValueError("Built-in provider profiles are read-only; create a custom profile instead.")
                connection.execute(
                    """
                    INSERT INTO provider_profiles(
                        id, name, provider_type, base_url, default_model, capabilities, secret_ref, is_builtin, enabled
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, FALSE, TRUE)
                    ON CONFLICT (id) DO UPDATE SET
                      name = EXCLUDED.name,
                      provider_type = EXCLUDED.provider_type,
                      base_url = EXCLUDED.base_url,
                      default_model = EXCLUDED.default_model,
                      capabilities = EXCLUDED.capabilities,
                      secret_ref = EXCLUDED.secret_ref,
                      updated_at = NOW()
                    """,
                    (
                        profile_uuid,
                        clean_name,
                        clean_type,
                        clean_url,
                        clean_model,
                        json.dumps(normalized_caps),
                        secret_ref,
                    ),
                )
        return ProviderProfile(
            str(profile_uuid), clean_name, clean_type, clean_url, clean_model, normalized_caps, secret_ref, False, True
        )


def _profile_from_row(row) -> ProviderProfile:
    capabilities = row["capabilities"]
    if isinstance(capabilities, str):
        capabilities = json.loads(capabilities)
    return ProviderProfile(
        str(row["id"]),
        str(row["name"]),
        str(row["provider_type"]),
        str(row["base_url"]),
        str(row["default_model"]),
        tuple(str(item) for item in (capabilities or ["text"])),
        str(row["secret_ref"]) if row["secret_ref"] else None,
        bool(row["is_builtin"]),
        bool(row["enabled"]),
    )


def profile_management_error(*, deployment_mode: str, role: str | None) -> str | None:
    mode = deployment_mode.strip().casefold()
    if mode not in {"local", "private", "public"}:
        return "KNOWLEDGELENS_DEPLOYMENT_MODE must be local, private, or public."
    if mode in {"local", "private"}:
        return None
    if role != "admin":
        return "Only an authenticated administrator may manage provider endpoints on a public deployment."
    return None


def configured_deployment_mode() -> str:
    return os.getenv("KNOWLEDGELENS_DEPLOYMENT_MODE", "local").strip().casefold()


def discover_models(profile: ProviderProfile, api_key: str = "") -> list[str]:
    endpoint = resolve_endpoint(profile.base_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    target = "/api/tags" if profile.provider_type == "ollama" else "/v1/models"
    status, body = get_pinned(endpoint, target, headers=headers)
    if status >= 400:
        raise RuntimeError(f"Provider model discovery returned HTTP {status}.")
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise RuntimeError("Provider model discovery returned invalid JSON.") from exc
    if profile.provider_type == "ollama":
        items = payload.get("models", []) if isinstance(payload, dict) else []
        values = [item.get("name") for item in items if isinstance(item, dict)]
    else:
        items = payload.get("data", []) if isinstance(payload, dict) else []
        values = [item.get("id") for item in items if isinstance(item, dict)]
    discovered = {
        str(value).strip()
        for value in values
        if value and 0 < len(str(value).strip()) <= MAX_MODEL_NAME_CHARS
    }
    return sorted(discovered)[:MAX_DISCOVERED_MODELS]


def legacy_profiles() -> list[ProviderProfile]:
    return list(LEGACY_PROFILES)
