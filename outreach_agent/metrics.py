"""Per-step timing and token accounting for a run.

Every graph node is wrapped by `instrument`, which times it and counts the model calls and tokens
made while it runs (including calls in tasks it starts). Each node appends one row to State.metrics.
"""

import inspect
import time

from .clients import current_usage
from .config import Settings

# Nodes that run after you approve contacts: the wait you actually feel.
AFTER_REVIEW = {"research_person", "verify_emails", "draft", "check", "save"}


def _new_usage() -> dict:
    return {"llm_calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}


def instrument(name: str, fn):
    def row(started: float, usage: dict) -> dict:
        return {"node": name, "start": started, "seconds": round(time.time() - started, 3), **usage}

    def merge(update, started, usage):
        if isinstance(update, dict):
            return {**update, "metrics": [row(started, usage)]}
        return update

    if inspect.iscoroutinefunction(fn):
        async def wrapped(state):
            started, usage = time.time(), _new_usage()
            token = current_usage.set(usage)
            try:
                return merge(await fn(state), started, usage)
            finally:
                current_usage.reset(token)
    else:
        def wrapped(state):
            started, usage = time.time(), _new_usage()
            token = current_usage.set(usage)
            try:
                return merge(fn(state), started, usage)
            finally:
                current_usage.reset(token)
    return wrapped


def cost(usage: dict, settings: Settings) -> float | None:
    """Estimated dollars, only if prices are configured (prices change; set them in .env)."""
    p_in, p_cached, p_out = settings.price_input, settings.price_cached_input, settings.price_output
    if p_in is None or p_out is None:
        return None
    cached = usage["cached_input_tokens"]
    fresh = usage["input_tokens"] - cached
    p_cached = p_in if p_cached is None else p_cached
    return (fresh * p_in + cached * p_cached + usage["output_tokens"] * p_out) / 1_000_000


def summarize(metrics: list[dict], settings: Settings | None = None) -> dict:
    """Per-node rows (repeated nodes such as a redraft are added up) and run totals."""
    by_node: dict[str, dict] = {}
    for m in metrics:
        agg = by_node.setdefault(m["node"], {"node": m["node"], "runs": 0, "seconds": 0.0, **_new_usage()})
        agg["runs"] += 1
        agg["seconds"] = round(agg["seconds"] + m["seconds"], 3)
        for k in _new_usage():
            agg[k] += m.get(k, 0)
    totals = _new_usage()
    for agg in by_node.values():
        for k in totals:
            totals[k] += agg[k]
    after = [m for m in metrics if m["node"] in AFTER_REVIEW]
    wait = (max(m["start"] + m["seconds"] for m in after) - min(m["start"] for m in after)) if after else 0.0
    return {"nodes": list(by_node.values()), "totals": totals, "wait_after_review": round(wait, 2),
            "cost": cost(totals, settings) if settings else None}


def to_markdown(summary: dict) -> str:
    t = summary["totals"]
    cached_share = f"{t['cached_input_tokens'] / t['input_tokens']:.0%}" if t["input_tokens"] else "0%"
    lines = ["## Run stats", "",
             f"- Wait after you approved contacts: **{summary['wait_after_review']:.1f} s**",
             f"- Model calls: {t['llm_calls']} · input tokens: {t['input_tokens']:,} "
             f"({cached_share} served from OpenAI's prompt cache) · output tokens: {t['output_tokens']:,}"]
    if summary.get("cost") is not None:
        lines.append(f"- Estimated model cost: ${summary['cost']:.4f}")
    lines += ["", "| Step | Runs | Seconds (summed) | Model calls | Input tokens | Cached | Output tokens |",
              "|---|---|---|---|---|---|---|"]
    for n in summary["nodes"]:
        lines.append(f"| {n['node']} | {n['runs']} | {n['seconds']:.2f} | {n['llm_calls']} | "
                     f"{n['input_tokens']:,} | {n['cached_input_tokens']:,} | {n['output_tokens']:,} |")
    lines.append("")
    lines.append("_Parallel steps overlap, so summed seconds exceed the wait._")
    return "\n".join(lines)


def one_line(summary: dict) -> str:
    t = summary["totals"]
    cached = f"{t['cached_input_tokens'] / t['input_tokens']:.0%}" if t["input_tokens"] else "0%"
    line = (f"Wait after approval: {summary['wait_after_review']:.1f} s · {t['llm_calls']} model calls · "
            f"{t['input_tokens'] + t['output_tokens']:,} tokens ({cached} of input from prompt cache)")
    if summary.get("cost") is not None:
        line += f" · ~${summary['cost']:.4f}"
    return line
