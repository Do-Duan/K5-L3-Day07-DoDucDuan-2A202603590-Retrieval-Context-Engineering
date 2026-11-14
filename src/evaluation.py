"""Gold evidence retrieval metrics and serializable per-query traces."""
from typing import Any
from collections import defaultdict


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """TODO: check source/page and enough gold evidence; keep hit@1 and hit@3 trace.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    gold = question["evidence_quote"]
    def hit(item):
        return (item["document_id"] == question["gold_source_id"] and
                int(item["position"]) == int(question["gold_position"]) and gold in item["content"])
    flags = [hit(h) for h in hits]
    return {"question_id": question["id"], "split": question["split"],
            "hit_at_1": bool(flags and flags[0]), "evidence_hit_at_3": any(flags),
            "hits": hits, "gold_position": question["gold_position"]}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """TODO: aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    groups = defaultdict(list)
    for row in traces:
        groups[(row["strategy"], row["split"])].append(row)
    return {f"{strategy}/{split}": {"n": len(rows),
            "hit_at_1": sum(r["hit_at_1"] for r in rows),
            "evidence_hit_at_3": sum(r["evidence_hit_at_3"] for r in rows)}
            for (strategy, split), rows in sorted(groups.items())}
