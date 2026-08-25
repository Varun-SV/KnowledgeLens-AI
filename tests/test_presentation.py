from knowledgelens.presentation import safe_tooltip_text


def test_tooltip_text_escapes_html_injection_payloads():
    raw = '<a href="https://evil.invalid" onmouseover="alert(1)"><img src=x onerror=alert(2)></a>'
    escaped = safe_tooltip_text(raw)

    assert "<a" not in escaped
    assert "<img" not in escaped
    assert "onerror=alert(2)>" not in escaped
    assert "&lt;a href=&quot;" in escaped
    assert "&lt;img" in escaped
