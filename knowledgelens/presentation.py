from __future__ import annotations

import html

from .limits import MAX_VISUALIZATION_EDGES, MAX_VISUALIZATION_NODES


def safe_tooltip_text(value: object) -> str:
    """Escape untrusted graph text for PyVis tooltip HTML fallback paths."""
    return html.escape(str(value), quote=True)


def parallel_edge_smooth(index: int, total: int) -> dict[str, object]:
    """Return a deterministic unique curve for each parallel edge in one endpoint pair."""
    if total <= 1:
        return {"enabled": True, "type": "continuous"}
    if index < 0 or index >= total:
        raise ValueError("Parallel-edge index must be within the endpoint-pair edge count.")

    clockwise = index % 2 == 0
    rank_on_side = index // 2
    side_count = (total + 1) // 2 if clockwise else total // 2

    # Spread each side's curves through a broad safe range instead of capping at a
    # fixed roundness. The curve type plus roundness pair therefore stays unique for
    # every edge all the way up to the supported visualization edge budget.
    roundness = 0.06 + (rank_on_side + 1) * (0.88 / (side_count + 1))
    return {
        "enabled": True,
        "type": "curvedCW" if clockwise else "curvedCCW",
        "roundness": roundness,
    }


def visualization_limit_error(node_count: int, edge_count: int) -> str | None:
    """Keep expensive force-directed rendering below a conservative interactive budget."""
    if node_count > MAX_VISUALIZATION_NODES or edge_count > MAX_VISUALIZATION_EDGES:
        return (
            "Interactive graph rendering is disabled for this workspace because it exceeds the safe visualization "
            f"budget ({MAX_VISUALIZATION_NODES:,} nodes / {MAX_VISUALIZATION_EDGES:,} edges). "
            "Graph chat and exports remain available."
        )
    return None
