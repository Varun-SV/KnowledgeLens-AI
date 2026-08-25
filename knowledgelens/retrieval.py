from __future__ import annotations

from difflib import SequenceMatcher

import networkx as nx

from .graph import is_auditable_claim_data
from .parsing import canonicalize_label

_RETRIEVAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)


def _token_list(value: str) -> list[str]:
    """Tokenize with the same identifier semantics used by graph canonicalization."""
    canonical = canonicalize_label(value)
    return canonical.split() if canonical else []


def _content_token_set(value: str) -> set[str]:
    """Return overlap tokens without generic grammatical words."""
    return {token for token in _token_list(value) if token not in _RETRIEVAL_STOPWORDS}


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

    # Stopwords remain available to exact phrase identity above, but they do not
    # create a semantic overlap match by themselves (for example, `the`).
    q_tokens = _content_token_set(q)
    n_tokens = _content_token_set(n)
    if q_tokens and n_tokens:
        overlap = len(q_tokens & n_tokens) / len(n_tokens)
        score += overlap * 2.0

    score += SequenceMatcher(None, q, n).ratio() * 0.75
    return score


def _is_evidentiary(data: dict) -> bool:
    return is_auditable_claim_data(data)


def _evidentiary_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    simple = nx.Graph()
    simple.add_edges_from(
        (subject, obj)
        for subject, obj, data in graph.edges(data=True)
        if _is_evidentiary(data)
    )
    return simple


def _master_evidence_neighbors(
    graph: nx.MultiDiGraph,
    master: str,
    evidence_graph: nx.Graph,
    limit: int,
) -> list[str]:
    """Use synthetic master topology only to choose nearby evidence-bearing seeds."""
    candidates: set[str] = set()
    for _subject, obj, data in graph.out_edges(master, data=True):
        if data.get("synthetic") and obj in evidence_graph and evidence_graph.degree(obj) > 0:
            candidates.add(str(obj))
    for subject, _obj, data in graph.in_edges(master, data=True):
        if data.get("synthetic") and subject in evidence_graph and evidence_graph.degree(subject) > 0:
            candidates.add(str(subject))

    return sorted(
        candidates,
        key=lambda node: (evidence_graph.degree(node), node.casefold()),
        reverse=True,
    )[:limit]


def relevant_nodes(graph: nx.MultiDiGraph, query: str, limit: int = 5) -> list[str]:
    ranked = sorted(
        ((score_node(query, str(node)), str(node)) for node in graph.nodes),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = [node for score, node in ranked if score >= 0.6]
    evidence_graph = _evidentiary_graph(graph)

    if selected:
        # Reserve the seed budget for directly matched evidence-bearing entities
        # before topology-only master expansion. Do not truncate the ranked matches
        # until after masters have been separated, otherwise a master can consume a
        # slot and push a lower-ranked explicit entity out before reservation occurs.
        protected: list[str] = []
        topology_only_masters: list[str] = []
        for node in selected:
            is_master = graph.nodes[node].get("type") == "master"
            has_direct_evidence = node in evidence_graph and evidence_graph.degree(node) > 0
            if is_master and not has_direct_evidence:
                topology_only_masters.append(node)
            elif node not in protected:
                protected.append(node)

        expanded = protected[:limit]
        for master in topology_only_masters:
            if len(expanded) >= limit:
                break
            neighbors = _master_evidence_neighbors(graph, master, evidence_graph, limit - len(expanded))
            for neighbor in neighbors:
                if neighbor not in expanded:
                    expanded.append(neighbor)
                if len(expanded) >= limit:
                    break

        if expanded:
            return expanded[:limit]

    # Generic queries should fall back to actual evidence, not a master node whose
    # only incident edges may be synthetic overview topology.
    evidence_nodes = sorted(evidence_graph.nodes, key=evidence_graph.degree, reverse=True)
    if evidence_nodes:
        return [str(evidence_nodes[0])]

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
    return f"[{source} · {location}{confidence_text}] {subject} --[{relation}]--> {obj}{evidence_text}"


def _claims_between(
    graph: nx.MultiDiGraph,
    left: str,
    right: str,
) -> list[tuple[tuple[str, str, str], str]]:
    claims: list[tuple[tuple[str, str, str], str]] = []
    for subject, obj in ((left, right), (right, left)):
        for key, data in graph.get_edge_data(subject, obj, default={}).items():
            if not _is_evidentiary(data):
                continue
            marker = (str(subject), str(obj), str(key))
            claims.append((marker, _claim_line(str(subject), str(obj), data)))
    return claims


def _incident_claims(graph: nx.MultiDiGraph, node: str) -> list[tuple[tuple[str, str, str], str]]:
    claims: list[tuple[tuple[str, str, str], str]] = []
    for subject, obj, key, data in graph.out_edges(node, keys=True, data=True):
        if _is_evidentiary(data):
            marker = (str(subject), str(obj), str(key))
            claims.append((marker, _claim_line(str(subject), str(obj), data)))
    for subject, obj, key, data in graph.in_edges(node, keys=True, data=True):
        if _is_evidentiary(data):
            marker = (str(subject), str(obj), str(key))
            claims.append((marker, _claim_line(str(subject), str(obj), data)))
    return claims


def _append_complete_line(lines: list[str], line: str, used_chars: int, max_chars: int) -> tuple[int, bool]:
    extra = len(line) + (1 if lines else 0)
    if used_chars + extra > max_chars:
        return used_chars, False
    lines.append(line)
    return used_chars + extra, True


def retrieve_graph_context(graph: nx.MultiDiGraph, query: str, max_chars: int = 9000) -> str:
    if graph.number_of_nodes() == 0:
        return "No graph context is available."

    seeds = relevant_nodes(graph, query)
    simple = _evidentiary_graph(graph)
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    used_chars = 0

    # Connecting paths are the highest-value context for multi-entity questions, so
    # reserve the front of the budget for them before any high-degree neighborhood
    # can consume the entire window.
    if len(seeds) >= 2:
        for index, source in enumerate(seeds):
            for target in seeds[index + 1 :]:
                try:
                    path = nx.shortest_path(simple, source, target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if not 1 < len(path) <= 6:
                    continue

                # A path header is topology metadata, not a citation. It deliberately
                # avoids bracket syntax so the answering model cannot confuse it with
                # source/location claim citations. Emit it only when at least one
                # source-backed claim for every hop also fits atomically.
                path_line = f"Graph path: {' -- '.join(map(str, path))}"
                required_claims: list[tuple[tuple[str, str, str], str]] = []
                path_is_supported = True
                for left, right in zip(path, path[1:], strict=False):
                    hop_claims = [
                        (marker, claim_line)
                        for marker, claim_line in _claims_between(graph, str(left), str(right))
                        if marker not in seen
                    ]
                    if not hop_claims:
                        path_is_supported = False
                        break
                    required_claims.append(hop_claims[0])

                if not path_is_supported:
                    continue

                bundle_lines = [path_line, *(claim_line for _marker, claim_line in required_claims)]
                bundle_extra = sum(len(line) for line in bundle_lines) + len(bundle_lines) - 1
                if lines:
                    bundle_extra += 1
                if used_chars + bundle_extra > max_chars:
                    continue

                used_chars, _ = _append_complete_line(lines, path_line, used_chars, max_chars)
                for marker, claim_line in required_claims:
                    used_chars, appended = _append_complete_line(lines, claim_line, used_chars, max_chars)
                    if appended:
                        seen.add(marker)

                # After the minimally supported path bundle is committed atomically,
                # include additional parallel claims for its hops if budget remains.
                for left, right in zip(path, path[1:], strict=False):
                    for marker, claim_line in _claims_between(graph, str(left), str(right)):
                        if marker in seen:
                            continue
                        next_used, appended = _append_complete_line(lines, claim_line, used_chars, max_chars)
                        if appended:
                            used_chars = next_used
                            seen.add(marker)

    # Then round-robin seed neighborhoods so one very high-degree entity cannot
    # starve the remaining selected entities. Never cut a claim line mid-citation.
    buckets = [_incident_claims(graph, node) for node in seeds]
    positions = [0] * len(buckets)
    while True:
        had_candidate = False
        for bucket_index, bucket in enumerate(buckets):
            while positions[bucket_index] < len(bucket):
                marker, claim_line = bucket[positions[bucket_index]]
                positions[bucket_index] += 1
                if marker in seen:
                    continue
                had_candidate = True
                next_used, appended = _append_complete_line(lines, claim_line, used_chars, max_chars)
                if appended:
                    used_chars = next_used
                    seen.add(marker)
                break
        if not had_candidate:
            break

    if not lines:
        return "No specific source-backed graph connections were found for the query."

    return "\n".join(lines)
