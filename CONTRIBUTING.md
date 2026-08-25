# Contributing to KnowledgeLens AI

Thanks for helping make KnowledgeLens more useful and more trustworthy.

## Before opening a pull request

1. Create a focused branch from the latest `main`.
2. Install development dependencies with `pip install -e ".[dev]"`.
3. Add or update tests for behavior changes.
4. Run:

```bash
python -m pytest
ruff check .
python -m compileall KnowledgeLens_AI.py knowledgelens
```

## Design principles

KnowledgeLens should prefer **inspectability over magic**. Features that extract or infer knowledge should preserve provenance whenever possible. Do not silently merge conflicting claims merely because their subject/object pair is the same.

Keep local and cloud providers equally viable. Any server-side network feature must consider SSRF, credential forwarding, redirects, and private-network exposure.

## Pull request notes

Please describe:

- what user problem the change solves;
- how the behavior was tested;
- any graph schema or saved-state compatibility impact;
- any privacy/security impact;
- screenshots for visible UI changes.
