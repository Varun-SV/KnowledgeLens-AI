from __future__ import annotations

import html


def safe_tooltip_text(value: object) -> str:
    """Escape untrusted graph text for PyVis tooltip HTML fallback paths."""
    return html.escape(str(value), quote=True)
