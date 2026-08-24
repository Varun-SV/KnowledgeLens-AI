import socket

import pytest

from knowledgelens.security import EndpointPolicyError, env_flag, resolve_endpoint, validate_endpoint


def test_blocks_localhost_by_default(monkeypatch):
    monkeypatch.delenv("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS", raising=False)
    with pytest.raises(EndpointPolicyError):
        validate_endpoint("http://localhost:11434")


def test_allows_localhost_with_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS", "1")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )
    assert validate_endpoint("http://localhost:11434") == "http://localhost:11434"


def test_falsey_local_flag_is_not_enabled(monkeypatch):
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS", value)
        assert env_flag("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS") is False


def test_blocks_private_resolved_address(monkeypatch):
    monkeypatch.delenv("KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS", raising=False)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 0))],
    )
    with pytest.raises(EndpointPolicyError):
        validate_endpoint("https://internal.example")


def test_shared_cgnat_space_requires_private_opt_in(monkeypatch):
    monkeypatch.delenv("KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS", raising=False)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.0.8", 0))],
    )
    with pytest.raises(EndpointPolicyError, match="non-global"):
        validate_endpoint("https://shared.example")

    monkeypatch.setenv("KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS", "1")
    assert validate_endpoint("https://shared.example") == "https://shared.example"


def test_allows_public_https(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    assert validate_endpoint("https://api.example.com") == "https://api.example.com"


def test_resolve_endpoint_retains_the_addresses_that_passed_policy(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", 0)),
        ],
    )
    endpoint = resolve_endpoint("https://api.example.com/base")
    assert [str(address) for address in endpoint.addresses] == ["93.184.216.34", "93.184.216.35"]
    assert endpoint.hostname == "api.example.com"
    assert endpoint.base_path == "/base"


def test_public_http_stays_blocked_when_local_opt_in_is_enabled(monkeypatch):
    monkeypatch.setenv("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS", "1")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    with pytest.raises(EndpointPolicyError, match="HTTPS"):
        validate_endpoint("http://api.example.com")
