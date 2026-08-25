from __future__ import annotations

import html

from .limits import MAX_VISUALIZATION_EDGES, MAX_VISUALIZATION_NODES


def safe_tooltip_text(value: object) -> str:
    """Escape untrusted graph text for PyVis tooltip HTML fallback paths."""
    return html.escape(str(value), quote=True)


def visualization_limit_error(node_count: int, edge_count: int) -> str | None:
    """Keep expensive force-directed rendering below a conservative interactive budget."""
    if node_count > MAX_VISUALIZATION_NODES or edge_count > MAX_VISUALIZATION_EDGES:
        return (
            "Interactive graph rendering is disabled for this workspace because it exceeds the safe visualization "
            f"budget ({MAX_VISUALIZATION_NODES:,} nodes / {MAX_VISUALIZATION_EDGES:,} edges). "
            "Graph chat and exports remain available."
        )
    return None
