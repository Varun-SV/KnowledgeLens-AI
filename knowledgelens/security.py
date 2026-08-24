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


def validate_endpoint(base_url: str) -> str:
    candidate = base_url.strip().rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise EndpointPolicyError("Use a full http:// or https:// endpoint URL.")
    if parsed.username or parsed.password:
        raise EndpointPolicyError("Credentials must not be embedded in the endpoint URL.")

    hostname = parsed.hostname.lower()
    allow_local = _flag("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS", default=False)
    allow_private = _flag("KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS", default=False)

    if hostname in {"localhost", "localhost.localdomain"}:
        if not allow_local:
            raise EndpointPolicyError(
                "Local endpoints are disabled by default. Set KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS=1 "
                "when running KnowledgeLens on your own machine."
            )
        return candidate

    addresses: list[ipaddress._BaseAddress] = []
    try:
        addresses.append(ipaddress.ip_address(hostname))
    except ValueError:
        try:
            for result in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM):
                addresses.append(ipaddress.ip_address(result[4][0]))
        except socket.gaierror as exc:
            raise EndpointPolicyError(f"Could not resolve endpoint host: {hostname}") from exc

    for address in addresses:
        if address.is_loopback:
            if not allow_local:
                raise EndpointPolicyError(
                    "Loopback endpoints are disabled by default. Set KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS=1 "
                    "for a trusted local deployment."
                )
        elif address.is_private or address.is_link_local or address.is_reserved or address.is_multicast:
            if not allow_private:
                raise EndpointPolicyError(
                    "Private or link-local endpoints are blocked by default. Set "
                    "KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS=1 only on a trusted self-hosted deployment."
                )

    if parsed.scheme == "http" and not (allow_local or allow_private):
        raise EndpointPolicyError("Public endpoints must use HTTPS.")

    return candidate
