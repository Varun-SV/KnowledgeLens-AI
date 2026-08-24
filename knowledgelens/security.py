from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse


class EndpointPolicyError(ValueError):
    pass


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_addresses(hostname: str) -> list[ipaddress._BaseAddress]:
    addresses: list[ipaddress._BaseAddress] = []
    try:
        addresses.append(ipaddress.ip_address(hostname))
    except ValueError:
        try:
            for result in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM):
                address = ipaddress.ip_address(result[4][0])
                if address not in addresses:
                    addresses.append(address)
        except socket.gaierror as exc:
            raise EndpointPolicyError(f"Could not resolve endpoint host: {hostname}") from exc
    if not addresses:
        raise EndpointPolicyError(f"Could not resolve endpoint host: {hostname}")
    return addresses


def validate_endpoint(base_url: str) -> str:
    candidate = base_url.strip().rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise EndpointPolicyError("Use a full http:// or https:// endpoint URL.")
    if parsed.username or parsed.password:
        raise EndpointPolicyError("Credentials must not be embedded in the endpoint URL.")
    if parsed.query or parsed.fragment:
        raise EndpointPolicyError("Endpoint URLs must not contain query strings or fragments.")

    hostname = parsed.hostname.casefold()
    allow_local = _flag("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS", default=False)
    allow_private = _flag("KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS", default=False)

    addresses = _resolve_addresses(hostname)
    http_safe = True

    for address in addresses:
        if address.is_unspecified or address.is_multicast or address.is_reserved:
            raise EndpointPolicyError("Unspecified, multicast, and reserved endpoint addresses are not allowed.")

        if address.is_loopback:
            if not allow_local:
                raise EndpointPolicyError(
                    "Loopback endpoints are disabled by default. Set KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS=1 "
                    "for a trusted local deployment."
                )
            continue

        if address.is_private or address.is_link_local:
            if not allow_private:
                raise EndpointPolicyError(
                    "Private or link-local endpoints are blocked by default. Set "
                    "KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS=1 only on a trusted self-hosted deployment."
                )
            continue

        # Any public address makes plaintext HTTP unsafe, regardless of local/private opt-ins.
        http_safe = False

    if parsed.scheme == "http" and not http_safe:
        raise EndpointPolicyError("Public endpoints must use HTTPS.")

    return candidate
