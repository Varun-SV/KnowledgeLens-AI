import ipaddress

from knowledgelens import http_client
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


def test_https_request_uses_validated_ip_and_original_tls_identity(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        http_client.urllib3,
        "HTTPSConnectionPool",
        lambda **kwargs: _FakePool(captured, **kwargs),
    )

    endpoint = ValidatedEndpoint(
        base_url="https://rebind.example/api",
        scheme="https",
        hostname="rebind.example",
        port=443,
        host_header="rebind.example",
        base_path="/api",
        addresses=(ipaddress.ip_address("93.184.216.34"),),
    )

    status, body = http_client.post_json_pinned(endpoint, {"model": "test"}, {"Authorization": "Bearer secret"})

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
