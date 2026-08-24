from __future__ import annotations

import re
from difflib import SequenceMatcher

import networkx as nx


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 1}


def score_node(query: str, node: str) -> float:
    q = query.lower().strip()
    n = str(node).lower().strip()
    if not q or not n:
        return 0.0
    score = 0.0
    if n in q:
        score += 3.0
    q_tokens = _tokens(q)
    n_tokens = _tokens(n)
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
    source = data.get("source", "unknown source")
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

    simple = nx.DiGraph()
    simple.add_nodes_from(graph.nodes)
    simple.add_edges_from((u, v) for u, v in graph.edges())
    if len(seeds) >= 2:
        for i, source in enumerate(seeds):
            for target in seeds[i + 1 :]:
                for candidate in ((source, target), (target, source)):
                    try:
                        path = nx.shortest_path(simple, candidate[0], candidate[1])
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    if 1 < len(path) <= 6:
                        lines.append(f"[graph path] {' -> '.join(map(str, path))}")
                        for a, b in zip(path, path[1:]):
                            for key, data in graph.get_edge_data(a, b, default={}).items():
                                marker = (str(a), str(b), str(key))
                                if marker not in seen:
                                    lines.append(_claim_line(str(a), str(b), data))
                                    seen.add(marker)
                        break

    if not lines:
        return "No specific graph connections were found for the query."

    return "\n".join(lines)[:max_chars]
