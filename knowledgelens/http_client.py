from __future__ import annotations

import json
from typing import Any

import urllib3
from urllib3.util import Timeout

from .limits import MAX_REQUEST_HEADERS_BYTES
from .security import IPAddress, ValidatedEndpoint

_MAX_REQUEST_BYTES = 96 * 1024
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


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
            raise PinnedRequestError("Outbound LLM request headers must not contain line breaks.")
        total_bytes += len(name.encode("utf-8")) + len(value.encode("utf-8")) + 4
    if total_bytes > MAX_REQUEST_HEADERS_BYTES:
        raise PinnedRequestError(
            f"The LLM request headers exceeded the {MAX_REQUEST_HEADERS_BYTES // 1024} KiB safety limit."
        )


def post_json_pinned(
    endpoint: ValidatedEndpoint,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """POST bounded JSON to one of the exact IP addresses that passed endpoint validation.

    The TCP connection is made to a validated IP literal, while the original hostname
    is retained in Host and, for HTTPS, SNI/certificate verification. This prevents a
    second DNS lookup from changing the destination after policy validation.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(body) > _MAX_REQUEST_BYTES:
        raise PinnedRequestError("The LLM request exceeded the 96 KiB safety limit.")

    request_headers = dict(headers or {})
    request_headers.setdefault("Content-Type", "application/json")
    request_headers["Host"] = endpoint.host_header
    _validate_headers(request_headers)

    target = f"{endpoint.base_path}/v1/chat/completions" if endpoint.base_path else "/v1/chat/completions"
    errors: list[str] = []

    for address in endpoint.addresses:
        pool = _pool_for(endpoint, address)
        response = None
        try:
            response = pool.urlopen(
                "POST",
                target,
                body=body,
                headers=request_headers,
                redirect=False,
                retries=False,
                preload_content=False,
            )
            data = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(data) > _MAX_RESPONSE_BYTES:
                raise PinnedRequestError("The LLM endpoint response exceeded the 5 MiB safety limit.")
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
    raise PinnedRequestError(f"Could not reach the LLM endpoint via its validated addresses: {detail}")
