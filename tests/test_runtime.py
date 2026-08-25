import pytest

from knowledgelens.runtime import (
    no_claims_build_error,
    provider_credential_error,
    provider_state_key,
    request_configuration_error,
)


def test_openai_requires_api_key_before_contacting_endpoint():
    assert provider_credential_error("OpenAI", "") is not None
    assert provider_credential_error("OpenAI", "   ") is not None
    assert provider_credential_error("OpenAI", "sk-test") is None
    assert provider_credential_error("Ollama / local", "") is None


def test_request_configuration_is_shared_by_build_and_chat_paths():
    assert request_configuration_error("OpenAI", "", "gpt-4o-mini") is not None
    assert request_configuration_error("OpenAI", "sk-test", "   ") == "Enter a model name."
    assert request_configuration_error("OpenAI", "sk-test", "gpt-4o-mini") is None
    assert request_configuration_error("Ollama / local", "", "llama3.1") is None


def test_provider_state_keys_are_distinct_so_credentials_cannot_bleed_between_providers():
    providers = ["Ollama / local", "llama.cpp / local", "OpenAI", "Configured endpoint"]
    keys = [provider_state_key(provider) for provider in providers]
    assert len(keys) == len(set(keys))
    with pytest.raises(ValueError, match="Unsupported provider"):
        provider_state_key("unknown")


def test_no_claims_build_error_is_only_returned_for_failed_extraction():
    error = no_claims_build_error(0, failures=3)
    assert error is not None
    assert "auditable claims" in error
    assert "3 chunk requests failed" in error
    assert no_claims_build_error(1, failures=3) is None
