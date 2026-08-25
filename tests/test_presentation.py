from knowledgelens.limits import MAX_VISUALIZATION_EDGES, MAX_VISUALIZATION_NODES
from knowledgelens.presentation import safe_tooltip_text, visualization_limit_error


def test_tooltip_text_escapes_html_injection_payloads():
    raw = '<a href="https://evil.invalid" onmouseover="alert(1)"><img src=x onerror=alert(2)></a>'
    escaped = safe_tooltip_text(raw)

    assert "<a" not in escaped
    assert "<img" not in escaped
    assert "onerror=alert(2)>" not in escaped
    assert "&lt;a href=&quot;" in escaped
    assert "&lt;img" in escaped


def test_visualization_budget_allows_boundary_and_blocks_oversize():
    assert visualization_limit_error(MAX_VISUALIZATION_NODES, MAX_VISUALIZATION_EDGES) is None

    node_error = visualization_limit_error(MAX_VISUALIZATION_NODES + 1, 0)
    edge_error = visualization_limit_error(1, MAX_VISUALIZATION_EDGES + 1)
    assert node_error is not None and "Graph chat and exports remain available" in node_error
    assert edge_error is not None and "Graph chat and exports remain available" in edge_error
