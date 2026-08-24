from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class EndpointPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedEndpoint:
    base_url: str
    scheme: str
    hostname: str
    port: int
    host_header: str
    base_path: str
    addresses: tuple[IPAddress, ...]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _resolve_addresses(hostname: str) -> list[IPAddress]:
    addresses: list[IPAddress] = []
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


def resolve_endpoint(base_url: str) -> ValidatedEndpoint:
    """Resolve and validate an endpoint, retaining the exact addresses that passed policy."""
    candidate = base_url.strip().rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise EndpointPolicyError("Use a full http:// or https:// endpoint URL.")
    if parsed.username or parsed.password:
        raise EndpointPolicyError("Credentials must not be embedded in the endpoint URL.")
    if parsed.query or parsed.fragment:
        raise EndpointPolicyError("Endpoint URLs must not contain query strings or fragments.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise EndpointPolicyError("Endpoint URL contains an invalid port.") from exc

    hostname = parsed.hostname.casefold()
    allow_local = env_flag("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS", default=False)
    allow_private = env_flag("KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS", default=False)

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

        # Treat every non-global address as internal/shared space. This includes
        # RFC1918, link-local, IPv6 local ranges, and 100.64.0.0/10 CGNAT/shared
        # address space, which ipaddress does not classify as `is_private`.
        if not address.is_global:
            if not allow_private:
                raise EndpointPolicyError(
                    "Private, link-local, shared, or otherwise non-global endpoints are blocked by default. "
                    "Set KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS=1 only on a trusted self-hosted deployment."
                )
            continue

        # Any globally routable address makes plaintext HTTP unsafe, regardless of opt-ins.
        http_safe = False

    if parsed.scheme == "http" and not http_safe:
        raise EndpointPolicyError("Public endpoints must use HTTPS.")

    return ValidatedEndpoint(
        base_url=candidate,
        scheme=parsed.scheme,
        hostname=hostname,
        port=port,
        host_header=parsed.netloc,
        base_path=parsed.path.rstrip("/"),
        addresses=tuple(addresses),
    )


def validate_endpoint(base_url: str) -> str:
    return resolve_endpoint(base_url).base_url
