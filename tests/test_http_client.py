import ipaddress

import pytest

from knowledgelens import http_client
from knowledgelens.limits import MAX_REQUEST_HEADERS_BYTES
from knowledgelens.security import ValidatedEndpoint


class _FakeResponse:
    status = 200

    def read(self, _amount):
        return b'{"choices":[{"message":{"content":"ok"}}]}'

    def release_conn(self):
        pass


class _FakePool:
    def __init__(self, captured, **kwargs):
        self.captured = captured
        captured["pool_kwargs"] = kwargs

    def urlopen(self, method, target, **kwargs):
        self.captured["method"] = method
        self.captured["target"] = target
        self.captured["request_kwargs"] = kwargs
        return _FakeResponse()

    def close(self):
        self.captured["closed"] = True


def _endpoint() -> ValidatedEndpoint:
    return ValidatedEndpoint(
        base_url="https://rebind.example/api",
        scheme="https",
        hostname="rebind.example",
        port=443,
        host_header="rebind.example",
        base_path="/api",
        addresses=(ipaddress.ip_address("93.184.216.34"),),
    )


def test_https_request_uses_validated_ip_and_original_tls_identity(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        http_client.urllib3,
        "HTTPSConnectionPool",
        lambda **kwargs: _FakePool(captured, **kwargs),
    )

    status, body = http_client.post_json_pinned(_endpoint(), {"model": "test"}, {"Authorization": "Bearer secret"})

    assert status == 200
    assert b'"content":"ok"' in body
    assert captured["pool_kwargs"]["host"] == "93.184.216.34"
    assert captured["pool_kwargs"]["server_hostname"] == "rebind.example"
    assert captured["pool_kwargs"]["assert_hostname"] == "rebind.example"
    assert captured["target"] == "/api/v1/chat/completions"
    assert captured["request_kwargs"]["headers"]["Host"] == "rebind.example"
    assert captured["request_kwargs"]["headers"]["Authorization"] == "Bearer secret"
    assert captured["request_kwargs"]["redirect"] is False
    assert captured["closed"] is True


def test_oversized_request_is_rejected_before_opening_connection(monkeypatch):
    def must_not_connect(*_args, **_kwargs):
        raise AssertionError("pool must not be constructed for oversized request")

    monkeypatch.setattr(http_client, "_pool_for", must_not_connect)
    payload = {"messages": [{"role": "user", "content": "x" * http_client._MAX_REQUEST_BYTES}]}

    with pytest.raises(http_client.PinnedRequestError, match="request exceeded"):
        http_client.post_json_pinned(_endpoint(), payload)


def test_oversized_or_multiline_headers_are_rejected_before_connection(monkeypatch):
    def must_not_connect(*_args, **_kwargs):
        raise AssertionError("pool must not be constructed for unsafe headers")

    monkeypatch.setattr(http_client, "_pool_for", must_not_connect)

    with pytest.raises(http_client.PinnedRequestError, match="headers exceeded"):
        http_client.post_json_pinned(
            _endpoint(),
            {"model": "test"},
            {"Authorization": "x" * MAX_REQUEST_HEADERS_BYTES},
        )

    with pytest.raises(http_client.PinnedRequestError, match="line breaks"):
        http_client.post_json_pinned(
            _endpoint(),
            {"model": "test"},
            {"Authorization": "Bearer safe\r\nX-Evil: injected"},
        )
