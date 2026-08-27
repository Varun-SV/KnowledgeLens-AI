from __future__ import annotations

import json
from typing import Any

import urllib3
from urllib3.util import Timeout

from .limits import MAX_REQUEST_HEADERS_BYTES
from .provider_activation import active_profile_request_config, restore_active_profile_environment
from .security import IPAddress, ValidatedEndpoint

_MAX_REQUEST_BYTES = 96 * 1024
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024

# The workspace already imports this transport before it builds the provider selector.
# Restoring the active profile here keeps the v0.2 UI compatible while allowing a
# PostgreSQL-backed endpoint to survive process restarts.
restore_active_profile_environment()


class PinnedRequestError(RuntimeError):
    pass


def _pool_for(endpoint: ValidatedEndpoint, address: IPAddress):
    common = {
        "host": str(address),
        "port": endpoint.port,
        "timeout": Timeout(connect=10.0, read=180.0),
        "retries": False,
        "maxsize": 1,
        "block": True,
    }
    if endpoint.scheme == "https":
        return urllib3.HTTPSConnectionPool(
            **common,
            cert_reqs="CERT_REQUIRED",
            assert_hostname=endpoint.hostname,
            server_hostname=endpoint.hostname,
        )
    return urllib3.HTTPConnectionPool(**common)


def _validate_headers(headers: dict[str, str]) -> None:
    total_bytes = 0
    for name, value in headers.items():
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            raise PinnedRequestError("Outbound request headers must not contain line breaks.")
        total_bytes += len(name.encode("utf-8")) + len(value.encode("utf-8")) + 4
    if total_bytes > MAX_REQUEST_HEADERS_BYTES:
        raise PinnedRequestError(f"The request headers exceeded the {MAX_REQUEST_HEADERS_BYTES // 1024} KiB safety limit.")


def _request_pinned(
    endpoint: ValidatedEndpoint,
    method: str,
    target: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    if not target.startswith("/") or "://" in target or "\r" in target or "\n" in target:
        raise PinnedRequestError("Pinned request targets must be relative absolute paths.")
    request_headers = dict(headers or {})
    request_headers["Host"] = endpoint.host_header
    _validate_headers(request_headers)
    errors: list[str] = []
    for address in endpoint.addresses:
        pool = _pool_for(endpoint, address)
        response = None
        try:
            response = pool.urlopen(
                method,
                target,
                body=body,
                headers=request_headers,
                redirect=False,
                retries=False,
                preload_content=False,
            )
            data = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(data) > _MAX_RESPONSE_BYTES:
                raise PinnedRequestError("The endpoint response exceeded the 5 MiB safety limit.")
            return int(response.status), data
        except PinnedRequestError:
            raise
        except (urllib3.exceptions.HTTPError, OSError) as exc:
            errors.append(f"{address}: {exc}")
        finally:
            if response is not None:
                response.release_conn()
            pool.close()
    detail = "; ".join(errors[:3]) or "no validated address accepted the connection"
    raise PinnedRequestError(f"Could not reach the endpoint via its validated addresses: {detail}")


def post_json_pinned(
    endpoint: ValidatedEndpoint,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """POST bounded LLM JSON to one of the exact IP addresses that passed endpoint validation.

    When the endpoint is the active persistent provider profile, the profile's saved
    model and credential are authoritative. This makes profile activation survive a
    full app-process restart while the legacy Streamlit provider controls remain
    available as a compatibility path.
    """
    request_payload = dict(payload)
    request_headers = dict(headers or {})
    active = active_profile_request_config(endpoint.base_url)
    if active:
        if "model" in request_payload:
            request_payload["model"] = active.model
        if active.secret and "Authorization" not in request_headers:
            request_headers["Authorization"] = f"Bearer {active.secret}"

    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    if len(body) > _MAX_REQUEST_BYTES:
        raise PinnedRequestError("The LLM request exceeded the 96 KiB safety limit.")
    request_headers.setdefault("Content-Type", "application/json")
    target = f"{endpoint.base_path}/v1/chat/completions" if endpoint.base_path else "/v1/chat/completions"
    return _request_pinned(endpoint, "POST", target, body=body, headers=request_headers)


def get_pinned(endpoint: ValidatedEndpoint, target: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    """GET a bounded relative path without re-resolving the validated hostname.

    This primitive never injects the active LLM credential. Provider discovery passes
    its credential explicitly, so future parser/storage callers cannot inherit an LLM
    secret merely because they share a host.
    """
    if endpoint.base_path and not target.startswith(endpoint.base_path + "/") and target != endpoint.base_path:
        target = f"{endpoint.base_path}{target}"
    return _request_pinned(endpoint, "GET", target, headers=headers)
