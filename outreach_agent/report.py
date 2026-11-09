"""Writes the drafts for review (Markdown) and appends contacts to a CSV tracker."""

import csv
from datetime import date
from pathlib import Path

from .domains import slug

TRACKER_FIELDS = ["date", "company", "role", "name", "title", "type", "email", "email_status", "linkedin", "status"]


def render_markdown(state: dict, today: date) -> str:
    research_by_name = {r["name"]: r for r in state.get("research", [])}
    lines = [f"# Outreach: {state['role']} at {state['company']}",
             f"_Generated {today}. Review and edit before sending._", ""]
    brief = state.get("role_brief") or {}
    if brief:
        lines += ["## About the role", brief.get("summary", ""), ""]
    for p in state["selected"]:
        d = state.get("drafts", {}).get(p["name"], {})
        r = research_by_name.get(p["name"], {})
        lines += [f"## {p['name']}, {p.get('position')} ({p['category'].replace('_', ' ')})",
                  f"- LinkedIn: {p.get('linkedin') or 'not found'}",
                  f"- Email: {p.get('email') or 'not found'} ({p.get('email_status') or 'unverified'})", ""]
        hooks = [f"- {h['fact']} ([source]({h['source_url']}))" for h in r.get("hooks", [])]
        lines += ["**Research**"] + (hooks or ["- Nothing specific found; message is based on the role and team."]) + [""]
        if d.get("edited"):
            lines += ["_Edited by you._", ""]
        lines += ["**LinkedIn note**", "```", d.get("linkedin_note", ""), "```",
                  f"**Email**: {d.get('email_subject', '')}", "```", d.get("email_body", ""), "```"]
        if state.get("feedback", {}).get(p["name"]):
            lines.append("> ⚠️ The checker still flagged: " + "; ".join(state["feedback"][p["name"]]))
        lines.append("")
    return "\n".join(lines)


def write_report(state: dict, outputs: Path, today: date | None = None, stats: dict | None = None) -> Path:
    today = today or date.today()
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f"{today}_{slug(state['company'])}_{slug(state['role'])}.md"
    text = render_markdown(state, today)
    if stats:
        from .metrics import to_markdown
        text += "\n" + to_markdown(stats) + "\n"
    path.write_text(text)

    tracker = outputs / "tracker.csv"
    new = not tracker.exists()
    with tracker.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(TRACKER_FIELDS)
        for p in state["selected"]:
            w.writerow([today, state["company"], state["role"], p["name"], p.get("position"), p["category"],
                        p.get("email"), p.get("email_status"), p.get("linkedin"), "drafted"])
    return path


def rewrite_report(state: dict, path: Path, today: date | None = None) -> Path:
    """Re-render an existing report (e.g. after you edit drafts) without touching the tracker.
    Keeps the run stats section that the first write appended."""
    old = path.read_text() if path.exists() else ""
    stats = old[old.index("## Run stats"):] if "## Run stats" in old else ""
    path.write_text(render_markdown(state, today or date.today()) + ("\n" + stats if stats else ""))
    return path
