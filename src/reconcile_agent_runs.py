"""Re-evaluate saved raw Agent answers locally after validator changes; no API calls."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import Agent


def reconcile(input_path: Path, output_path: Path) -> dict[str, Any]:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    reviewed: list[dict[str, Any]] = []
    for original in report["traces"]:
        trace = dict(original)
        if trace.get("status") != "error" and trace.get("model_answer") is not None:
            agent = Agent.__new__(Agent)
            agent.model = trace.get("model", "unknown")
            context = {item["id"]: item for item in trace.get("context", [])}
            prior_events = [event for event in trace.get("events", [])
                            if event.get("event") not in ("answer", "abstain")]
            result = agent._finish(
                trace["question"], trace["model_answer"], context, prior_events,
                time.perf_counter(), trace.get("search_count", 0),
            )
            preserved = {key: value for key, value in trace.items()
                         if key not in ("status", "answer", "citations", "events",
                                        "citation_validation_error")}
            trace = {**preserved, **result}
            trace["query_id"] = original["query_id"]
            trace["answerable"] = original["answerable"]
            trace["split"] = original["split"]
            trace["elapsed_ms"] = original.get("elapsed_ms")
            trace["reprocessed_locally"] = True
        reviewed.append(trace)

    answerable = [item for item in reviewed if item.get("answerable") is True]
    unanswerable = [item for item in reviewed if item.get("answerable") is False]
    successful = [item for item in reviewed if item.get("status") != "error"]
    summary = dict(report.get("summary", {}))
    summary.update({
        "answered": sum(item.get("status") == "answered" for item in successful),
        "abstained": sum(item.get("status") == "abstain" for item in successful),
        "answerable_answered": {"n": sum(item.get("status") == "answered" for item in answerable),
                                "N": len(answerable)},
        "answerable_abstained": {"n": sum(item.get("status") == "abstain" for item in answerable),
                                 "N": len(answerable)},
        "unanswerable_abstained": {"n": sum(item.get("status") == "abstain" for item in unanswerable),
                                   "N": len(unanswerable)},
        "status_review": "reprocessed locally from saved model outputs; no additional API calls",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "traces": reviewed},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn hóa trạng thái/citation từ trace đã lưu; không gọi API")
    parser.add_argument("--input", type=Path, default=ROOT / "src" / "results" / "agent-dev.json")
    parser.add_argument("--output", type=Path, default=ROOT / "src" / "results" / "agent-dev-reviewed.json")
    args = parser.parse_args()
    print(json.dumps(reconcile(args.input, args.output), ensure_ascii=False, indent=2))
    print(f"Reviewed trace saved: {args.output}")


if __name__ == "__main__":
    main()
