"""Measure how well the draft checker catches seeded errors, and how often it flags accurate drafts.

    python -m evals.run_checker_eval                 # one pass, ~42 small model calls (a few cents)
    python -m evals.run_checker_eval --repeats 3     # also measures how stable the verdicts are
    python -m evals.run_checker_eval --rules-only    # deterministic checks alone, free

Writes evals/results/<timestamp>/results.jsonl and summary.md.
"""

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from outreach_agent import prompts
from outreach_agent.checks import rule_problems
from outreach_agent.schemas import FactCheck

from .checker_dataset import COMPANY, build_cases

RESULTS_DIR = Path(__file__).parent / "results"


async def check_case(case: dict, llm, rules_only: bool) -> dict:
    rules = rule_problems(case["draft"], recipient_name=case["recipient"], company=COMPANY)
    model = []
    if not rules_only:
        verdict = await llm.parse([
            {"role": "system", "content": prompts.CHECK_PROMPT},
            {"role": "user", "content": json.dumps({
                "draft": case["draft"], "candidate_profile": case["candidate_profile"],
                "person_research": case["person_research"], "role_brief": case["role_brief"]})}], FactCheck)
        model = verdict.problems
    return {"id": case["id"], "error": case["error"], "flagged": bool(rules or model),
            "rule_problems": rules, "model_problems": model}


def summarize(rows: list[dict]) -> dict:
    by_kind = defaultdict(lambda: [0, 0])          # kind -> [flagged, total]
    for r in rows:
        kind = r["error"] or "clean"
        by_kind[kind][0] += r["flagged"]
        by_kind[kind][1] += 1
    seeded = [r for r in rows if r["error"]]
    clean = [r for r in rows if not r["error"]]
    return {
        "recall": sum(r["flagged"] for r in seeded) / len(seeded) if seeded else 0.0,
        "false_positive_rate": sum(r["flagged"] for r in clean) / len(clean) if clean else 0.0,
        "by_kind": {k: {"flagged": f, "total": t, "rate": f / t} for k, (f, t) in sorted(by_kind.items())},
    }


def stability(passes: list[list[dict]]) -> float:
    """Share of cases that got the same flagged/not-flagged verdict on every pass."""
    verdicts = defaultdict(set)
    for rows in passes:
        for r in rows:
            verdicts[r["id"]].add(r["flagged"])
    return sum(len(v) == 1 for v in verdicts.values()) / len(verdicts)


def to_markdown(summary: dict, model: str, repeats: int, stable: float | None) -> str:
    lines = [f"# Checker eval ({model}, {repeats} pass{'es' if repeats > 1 else ''})", "",
             f"- **Recall on seeded errors:** {summary['recall']:.0%}",
             f"- **False positive rate on accurate drafts:** {summary['false_positive_rate']:.0%}"]
    if stable is not None:
        lines.append(f"- **Same verdict on every pass:** {stable:.0%}")
    lines += ["", "| Case type | Flagged | Rate |", "|---|---|---|"]
    lines += [f"| {k} | {v['flagged']}/{v['total']} | {v['rate']:.0%} |" for k, v in summary["by_kind"].items()]
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()

    llm, model = None, "rules only"
    if not args.rules_only:
        from openai import AsyncOpenAI

        from outreach_agent.clients import OpenAIStructured
        from outreach_agent.config import Settings
        model = Settings().model
        llm = OpenAIStructured(AsyncOpenAI(), model)

    cases = build_cases()
    sem = asyncio.Semaphore(8)

    async def bounded(case):
        async with sem:
            return await check_case(case, llm, args.rules_only)

    passes = [await asyncio.gather(*(bounded(c) for c in cases)) for _ in range(args.repeats)]
    rows = [r for p in passes for r in p]
    summary = summarize(rows)
    stable = stability(passes) if args.repeats > 1 else None

    out = RESULTS_DIR / f"{datetime.now():%Y%m%dT%H%M%S}_{model.replace('/', '-').replace(' ', '-')}"
    out.mkdir(parents=True)
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report = to_markdown(summary, model, args.repeats, stable)
    (out / "summary.md").write_text(report)
    print(report)
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
