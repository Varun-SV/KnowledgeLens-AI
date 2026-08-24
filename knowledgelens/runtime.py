from __future__ import annotations


def provider_credential_error(provider: str, api_key: str) -> str | None:
    """Return a user-facing build validation error for provider credentials."""
    if provider == "OpenAI" and not api_key.strip():
        return "Enter an OpenAI API key before building the evidence graph."
    return None


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
