from __future__ import annotations

import json
import os
import uuid

from .database import Database, DatabaseUnavailable
from .secrets import build_secret_store

_ACTIVE_KEY = "active_provider_profile_id"


def _profile_uuid(profile_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(profile_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid provider profile identifier.") from exc


def set_active_profile(database: Database, profile_id: str) -> None:
    profile_uuid = _profile_uuid(profile_id)
    with database.connect() as connection:
        with connection.transaction():
            profile = connection.execute(
                "SELECT id FROM provider_profiles WHERE id = %s AND enabled = TRUE",
                (profile_uuid,),
            ).fetchone()
            if not profile:
                raise ValueError("Provider profile does not exist or is disabled.")
            connection.execute(
                """
                INSERT INTO app_settings(key, value_json) VALUES (%s, %s::jsonb)
                ON CONFLICT (key) DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = NOW()
                """,
                (_ACTIVE_KEY, json.dumps({"profile_id": str(profile_uuid)})),
            )
    restore_active_profile_environment(database)


def active_profile_record(database: Database) -> dict | None:
    if not database.enabled:
        return None
    with database.connect() as connection:
        setting = connection.execute("SELECT value_json FROM app_settings WHERE key = %s", (_ACTIVE_KEY,)).fetchone()
        if not setting:
            return None
        value = setting["value_json"]
        if isinstance(value, str):
            value = json.loads(value)
        profile_id = value.get("profile_id") if isinstance(value, dict) else None
        if not profile_id:
            return None
        try:
            profile_uuid = _profile_uuid(str(profile_id))
        except ValueError:
            return None
        row = connection.execute(
            """
            SELECT id, name, provider_type, base_url, default_model, secret_ref
            FROM provider_profiles WHERE id = %s AND enabled = TRUE
            """,
            (profile_uuid,),
        ).fetchone()
    return dict(row) if row else None


def restore_active_profile_environment(database: Database | None = None) -> dict | None:
    database = database or Database()
    try:
        profile = active_profile_record(database)
    except DatabaseUnavailable:
        return None
    if not profile:
        return None
    os.environ["KNOWLEDGELENS_CUSTOM_ENDPOINT"] = str(profile["base_url"]).rstrip("/")
    os.environ["KNOWLEDGELENS_CUSTOM_MODEL"] = str(profile["default_model"])
    os.environ["KNOWLEDGELENS_ACTIVE_PROVIDER_TYPE"] = str(profile["provider_type"])
    return profile


def active_profile_secret_for_endpoint(endpoint_base_url: str) -> str | None:
    database = Database()
    try:
        profile = active_profile_record(database)
    except DatabaseUnavailable:
        return None
    if not profile or not profile.get("secret_ref"):
        return None
    if str(profile["base_url"]).rstrip("/") != endpoint_base_url.rstrip("/"):
        return None
    try:
        return build_secret_store(database).get(str(profile["secret_ref"]))
    except RuntimeError:
        return None
