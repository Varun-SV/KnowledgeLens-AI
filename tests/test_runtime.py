from knowledgelens.runtime import no_claims_build_error, provider_credential_error


def test_openai_requires_api_key_before_building():
    assert provider_credential_error("OpenAI", "") is not None
    assert provider_credential_error("OpenAI", "   ") is not None
    assert provider_credential_error("OpenAI", "sk-test") is None
    assert provider_credential_error("Ollama / local", "") is None


def test_no_claims_build_error_is_only_returned_for_failed_extraction():
    error = no_claims_build_error(0, failures=3)
    assert error is not None
    assert "auditable claims" in error
    assert "3 chunk requests failed" in error
    assert no_claims_build_error(1, failures=3) is None
