from pathlib import Path


def test_extraction_prompt_requires_verbatim_source_evidence():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    assert '"evidence":"short verbatim source excerpt"' in app_source
    assert "Evidence MUST be a short verbatim excerpt copied from the supplied source text" in app_source
    assert "never paraphrase or invent evidence" in app_source
