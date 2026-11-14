"""Run the Agent over one benchmark split without exposing gold to the model."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from src.agent import Agent

DATA_PATH = ROOT / "src" / "data" / "retrieval_benchmark.json"


def run_benchmark(split: str, output_path: Path, *, model: str | None = None,
                  method: str = "hybrid", top_k: int = 5, max_searches: int = 3,
                  context_budget_chars: int = 6000,
                  questions_path: Path = DATA_PATH) -> dict[str, Any]:
    cases = json.loads(questions_path.read_text(encoding="utf-8"))
    selected = [case for case in cases if case.get("split") == split]
    if not selected:
        raise ValueError(f"Không có câu hỏi split={split} trong {questions_path}")
    if len({case.get("id") for case in selected}) != len(selected):
        raise ValueError("ID câu hỏi trùng trong split benchmark")

    agent = Agent(model, max_searches=max_searches, top_k=top_k,
                  context_budget_chars=context_budget_chars, default_method=method)
    traces: list[dict[str, Any]] = []
    for case in selected:
        # Only the natural-language question enters the Agent; gold/expected
        # answers remain in this evaluator and never enter prompts or retrieval.
        try:
            result = agent.run(case["question"])
            trace = {**result, "query_id": case["id"], "split": split,
                     "answerable": case["answerable"]}
        except Exception as exc:
            trace = {"query_id": case["id"], "split": split,
                     "answerable": case["answerable"], "question": case["question"],
                     "status": "error", "answer": None, "citations": [],
                     "model": agent.model, "search_count": 0, "context_ids": [],
                     "context": [], "events": [{"event": "error",
                     "error": f"{type(exc).__name__}: {exc}"}], "elapsed_ms": None}
        traces.append(trace)

    answerable = [trace for trace in traces if trace["answerable"]]
    unanswerable = [trace for trace in traces if not trace["answerable"]]
    successful = [trace for trace in traces if trace["status"] != "error"]
    search_calls = [trace["search_count"] for trace in successful]
    elapsed = [trace["elapsed_ms"] for trace in successful if trace.get("elapsed_ms") is not None]
    summary = {
        "split": split,
        "cases": len(traces),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "errors": sum(trace["status"] == "error" for trace in traces),
        "answered": sum(trace["status"] == "answered" for trace in successful),
        "abstained": sum(trace["status"] == "abstain" for trace in successful),
        "answerable_answered": {"n": sum(t["status"] == "answered" for t in answerable),
                                "N": len(answerable)},
        "answerable_abstained": {"n": sum(t["status"] == "abstain" for t in answerable),
                                 "N": len(answerable)},
        "unanswerable_abstained": {"n": sum(t["status"] == "abstain" for t in unanswerable),
                                   "N": len(unanswerable)},
        "tool_use": {"n": sum(t["search_count"] > 0 for t in successful), "N": len(successful)},
        "mean_searches": statistics.mean(search_calls) if search_calls else None,
        "median_elapsed_ms": statistics.median(elapsed) if elapsed else None,
        "model": agent.model,
        "retrieval_method_default": method,
        "top_k": top_k,
        "max_searches": max_searches,
        "context_budget_chars": context_budget_chars,
        "grading_note": "Status metrics only; manually grade answer grounding and citation support before reporting quality.",
    }
    report = {"summary": summary, "traces": traces}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Đánh giá Agent trên một split; không gửi gold vào Agent")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--questions", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--method", choices=("dense", "tfidf", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--context-budget-chars", type=int, default=6000)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_searches < 1 or args.context_budget_chars < 1:
        parser.error("top-k, max-searches và context-budget-chars phải >= 1")
    output = args.output or ROOT / "src" / "results" / f"agent-{args.split}.json"
    summary = run_benchmark(args.split, output, model=args.model, method=args.method,
                            top_k=args.top_k, max_searches=args.max_searches,
                            context_budget_chars=args.context_budget_chars,
                            questions_path=args.questions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Trace saved: {output}")


if __name__ == "__main__":
    main()
