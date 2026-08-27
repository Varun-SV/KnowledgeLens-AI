import pytest

from knowledgelens.auth import hash_password, verify_password
from knowledgelens.database import Database, DatabaseSettings
from knowledgelens.provider_profiles import ProviderProfileRepository


def test_provider_repository_rejects_unknown_provider_type_before_io():
    repository = ProviderProfileRepository(Database(DatabaseSettings(url=None)))
    with pytest.raises(ValueError, match="Unsupported provider"):
        repository.save(
            name="Invalid",
            provider_type="unsupported",
            base_url="invalid",
            default_model="model",
            capabilities=("text",),
        )


def test_password_verification_rejects_modified_work_factor():
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded)
    fields = encoded.split("$")
    fields[1] = "32768"
    assert not verify_password("correct horse battery staple", "$".join(fields))
