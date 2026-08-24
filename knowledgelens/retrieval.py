from __future__ import annotations

from difflib import SequenceMatcher

import networkx as nx


def _token_list(value: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in value.casefold():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _contains_token_phrase(query: str, node: str) -> bool:
    q_tokens = _token_list(query)
    n_tokens = _token_list(node)
    if not q_tokens or not n_tokens or len(n_tokens) > len(q_tokens):
        return False
    width = len(n_tokens)
    return any(q_tokens[index : index + width] == n_tokens for index in range(len(q_tokens) - width + 1))


def score_node(query: str, node: str) -> float:
    q = query.casefold().strip()
    n = str(node).casefold().strip()
    if not q or not n:
        return 0.0

    score = 0.0
    if _contains_token_phrase(q, n):
        score += 3.0

    q_tokens = set(_token_list(q))
    n_tokens = set(_token_list(n))
    if q_tokens and n_tokens:
        overlap = len(q_tokens & n_tokens) / len(n_tokens)
        score += overlap * 2.0

    score += SequenceMatcher(None, q, n).ratio() * 0.75
    return score


def relevant_nodes(graph: nx.MultiDiGraph, query: str, limit: int = 5) -> list[str]:
    ranked = sorted(
        ((score_node(query, str(node)), str(node)) for node in graph.nodes),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = [node for score, node in ranked if score >= 0.6][:limit]
    if selected:
        return selected

    masters = [node for node in graph.nodes if graph.nodes[node].get("type") == "master"]
    if masters:
        return [str(masters[0])]
    return [str(node) for node in sorted(graph.nodes, key=graph.degree, reverse=True)[:1]]


def _claim_line(subject: str, obj: str, data: dict) -> str:
    source = data.get("source") or "unknown source"
    legacy_sources = [str(item) for item in data.get("legacy_sources", []) if item]
    if source == "unknown source" and legacy_sources:
        source = "legacy candidates: " + ", ".join(legacy_sources)

    page = data.get("page")
    chunk = data.get("chunk_index")
    location = f"p.{page}" if page is not None else f"chunk {chunk}"
    relation = data.get("relation", "related to")
    evidence = " ".join(str(data.get("evidence") or "").split())
    confidence = data.get("confidence")
    confidence_text = f" · confidence {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
    evidence_text = f" · evidence: {evidence}" if evidence else ""
    synthetic = " · synthetic overview link" if data.get("synthetic") else ""
    return (
        f"[{source} · {location}{confidence_text}{synthetic}] "
        f"{subject} --[{relation}]--> {obj}{evidence_text}"
    )


def _append_claims_between(
    graph: nx.MultiDiGraph,
    left: str,
    right: str,
    lines: list[str],
    seen: set[tuple[str, str, str]],
) -> None:
    for subject, obj in ((left, right), (right, left)):
        for key, data in graph.get_edge_data(subject, obj, default={}).items():
            marker = (str(subject), str(obj), str(key))
            if marker not in seen:
                lines.append(_claim_line(str(subject), str(obj), data))
                seen.add(marker)


def retrieve_graph_context(graph: nx.MultiDiGraph, query: str, max_chars: int = 9000) -> str:
    if graph.number_of_nodes() == 0:
        return "No graph context is available."

    seeds = relevant_nodes(graph, query)
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for node in seeds:
        for subject, obj, key, data in graph.out_edges(node, keys=True, data=True):
            marker = (str(subject), str(obj), str(key))
            if marker not in seen:
                lines.append(_claim_line(str(subject), str(obj), data))
                seen.add(marker)
        for subject, obj, key, data in graph.in_edges(node, keys=True, data=True):
            marker = (str(subject), str(obj), str(key))
            if marker not in seen:
                lines.append(_claim_line(str(subject), str(obj), data))
                seen.add(marker)

    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes)
    simple.add_edges_from((u, v) for u, v in graph.edges())

    if len(seeds) >= 2:
        for index, source in enumerate(seeds):
            for target in seeds[index + 1 :]:
                try:
                    path = nx.shortest_path(simple, source, target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if 1 < len(path) <= 6:
                    lines.append(f"[graph path] {' -- '.join(map(str, path))}")
                    for left, right in zip(path, path[1:]):
                        _append_claims_between(graph, str(left), str(right), lines, seen)

    if not lines:
        return "No specific graph connections were found for the query."

    return "\n".join(lines)[:max_chars]
