from __future__ import annotations

_PROVIDER_STATE_KEYS = {
    "Ollama / local": "ollama",
    "llama.cpp / local": "llama_cpp",
    "OpenAI": "openai",
    "Configured endpoint": "configured",
}


def provider_state_key(provider: str) -> str:
    """Return a stable UI/session namespace for one supported provider."""
    try:
        return _PROVIDER_STATE_KEYS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def provider_credential_error(provider: str, api_key: str) -> str | None:
    """Return a user-facing validation error for provider credentials."""
    if provider == "OpenAI" and not api_key.strip():
        return "Enter an OpenAI API key before contacting the OpenAI endpoint."
    return None


def request_configuration_error(provider: str, api_key: str, model_name: str) -> str | None:
    """Validate request settings shared by build and graph-chat request paths."""
    if not model_name.strip():
        return "Enter a model name."
    return provider_credential_error(provider, api_key)


def no_claims_build_error(claim_count: int, failures: int) -> str | None:
    """Return a persistent build error when no auditable claims survived extraction."""
    if claim_count > 0:
        return None

    failure_detail = f" {failures} chunk requests failed." if failures else ""
    return (
        "The model did not produce any auditable claims with supporting evidence. "
        "Try a stronger instruction-following model, inspect the source text, or adjust the extraction focus."
        f"{failure_detail}"
    )
