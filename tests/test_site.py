from pathlib import Path


def test_landing_reveal_animation_fails_open_without_javascript_or_observer():
    index = Path("site/index.html").read_text(encoding="utf-8")
    script = Path("site/app.js").read_text(encoding="utf-8")

    assert "<noscript><style>.reveal{opacity:1!important;transform:none!important}</style></noscript>" in index
    assert "window.__klRevealAnimationReady = false" in index
    assert "if (!window.__klRevealAnimationReady)" in index
    assert "typeof IntersectionObserver === 'function'" in script
    assert "showAllReveals();" in script
    assert "window.__klRevealAnimationReady = true" in script
