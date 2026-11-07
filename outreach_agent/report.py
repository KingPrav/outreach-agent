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
        lines += ["**LinkedIn note**", "```", d.get("linkedin_note", ""), "```",
                  f"**Email**: {d.get('email_subject', '')}", "```", d.get("email_body", ""), "```"]
        if state.get("feedback", {}).get(p["name"]):
            lines.append("> ⚠️ The checker still flagged: " + "; ".join(state["feedback"][p["name"]]))
        lines.append("")
    return "\n".join(lines)


def write_report(state: dict, outputs: Path, today: date | None = None) -> Path:
    today = today or date.today()
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f"{today}_{slug(state['company'])}_{slug(state['role'])}.md"
    path.write_text(render_markdown(state, today))

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
