from __future__ import annotations

from dataclasses import dataclass

from .blob_store import LocalContentStore
from .database import Database, DatabaseUnavailable
from .provider_profiles import ProviderProfileRepository
from .secrets import SecretStore, build_secret_store


@dataclass(slots=True)
class ApplicationServices:
    database: Database
    blobs: LocalContentStore
    profiles: ProviderProfileRepository | None
    secrets: SecretStore | None
    persistence_error: str | None = None

    @property
    def persistent(self) -> bool:
        return self.profiles is not None and self.secrets is not None and not self.persistence_error


def build_application_services() -> ApplicationServices:
    database = Database()
    blobs = LocalContentStore()
    if not database.enabled:
        return ApplicationServices(database, blobs, None, None, "PostgreSQL is not configured; using legacy in-session provider settings.")
    try:
        database.initialize()
        repository = ProviderProfileRepository(database)
        repository.seed_builtins()
        return ApplicationServices(database, blobs, repository, build_secret_store(database))
    except DatabaseUnavailable as exc:
        return ApplicationServices(database, blobs, None, None, str(exc))
