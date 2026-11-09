"""Web UI (Gradio) for the outreach agent.

    pip install -e ".[ui]"
    python -m outreach_agent.ui          # opens http://127.0.0.1:7860

The UI drives the same graph as the command line. Each button opens the MCP servers, runs the
graph until it pauses or finishes, and closes them; the run's state lives in the SQLite
checkpointer between clicks, keyed by the run id. So a review can be resumed later, from the UI
or the command line.
"""

import csv
import html
from pathlib import Path
from urllib.parse import quote, urlencode
from contextlib import AbstractAsyncContextManager
from typing import Callable

from langgraph.types import Command

from .cli import new_run_id
from .clients import AgentError
from .config import LINKEDIN_NOTE_LIMIT, Settings
from .graph import build_graph
from .metrics import one_line, summarize
from .runtime import Services, open_services

ServicesFactory = Callable[[Settings], AbstractAsyncContextManager[Services]]


def contact_label(i: int, p: dict) -> str:
    return f"{i + 1}. {p['name']}, {p.get('position') or 'unknown title'} ({p['category'].replace('_', ' ')}): {p.get('why', '')}"


class OutreachRunner:
    """Gradio-free logic, so it can be tested with fake services."""

    def __init__(self, settings: Settings | None = None, services: ServicesFactory = open_services):
        self.settings = settings or Settings()
        self.services = services

    async def _run(self, run_id: str, payload) -> tuple[dict, list[str]]:
        logs: list[str] = []
        async with self.services(self.settings) as svc:
            app = build_graph(svc.people, svc.research, svc.llm, self.settings, svc.checkpointer,
                              log=lambda line: logs.append(line.strip()))
            config = {"configurable": {"thread_id": run_id}}
            if payload is None:                                   # just read a paused run
                snapshot = await app.aget_state(config)
                if not snapshot.tasks or not snapshot.tasks[0].interrupts:
                    raise AgentError(f"No paused run with id {run_id}.")
                return {"__interrupt__": snapshot.tasks[0].interrupts}, logs
            return await app.ainvoke(payload, config), logs

    async def start(self, company: str, role: str, job_url: str = "", domain: str = "") -> dict:
        if not company.strip() or not role.strip():
            raise AgentError("Enter a company and a role.")
        run_id = new_run_id(company, role)
        inputs = {"company": company.strip(), "role": role.strip(),
                  "job_url": job_url.strip(), "domain": domain.strip()}
        result, logs = await self._run(run_id, inputs)
        shortlist = result["__interrupt__"][0].value["shortlist"] if "__interrupt__" in result else []
        return {"run_id": run_id, "shortlist": shortlist, "logs": logs}

    async def load_paused(self, run_id: str) -> dict:
        result, logs = await self._run(run_id.strip(), None)
        return {"run_id": run_id.strip(), "shortlist": result["__interrupt__"][0].value["shortlist"], "logs": logs}

    async def approve(self, run_id: str, indices: list[int]) -> dict:
        result, logs = await self._run(run_id, Command(resume=indices))
        return {"result": result, "logs": logs}


EDIT_KEYS = ("linkedin_note", "email_subject", "email_body")


def current_text(result: dict, edits: dict, name: str) -> dict:
    """The text to show for a contact: your edits if you made any, otherwise the agent's draft."""
    draft = result.get("drafts", {}).get(name, {})
    return {k: edits.get(name, {}).get(k, draft.get(k, "")) for k in EDIT_KEYS}


def note_counter(note: str) -> str:
    n = len(note or "")
    return f"{n}/{LINKEDIN_NOTE_LIMIT} characters" + (" (LinkedIn will cut it off)" if n > LINKEDIN_NOTE_LIMIT else "")


def apply_edits(result: dict, edits: dict) -> dict:
    """A copy of the result with your edited text in the drafts, marked as edited."""
    drafts = {name: dict(d) for name, d in result.get("drafts", {}).items()}
    for name, changed in edits.items():
        if name in drafts and any(drafts[name].get(k) != v for k, v in changed.items()):
            drafts[name].update(changed, edited=True)
    return {**result, "drafts": drafts}


def draft_fields(result: dict, name: str, edits: dict | None = None) -> tuple[str, str, str, str]:
    """(LinkedIn note, subject, email body, notes markdown) for one contact."""
    d = current_text(result, edits or {}, name)
    research = next((r for r in result.get("research", []) if r["name"] == name), {})
    person = next((p for p in result.get("selected", []) if p["name"] == name), {})
    note = d.get("linkedin_note", "")
    sections = [f"**Email:** {person.get('email') or 'not found'} ({person.get('email_status') or 'unverified'})"]
    if person.get("linkedin"):
        sections.append(f"**LinkedIn:** {person['linkedin']}")
    hooks = [f"- {h['fact']} ([source]({h['source_url']}))" for h in research.get("hooks", [])]
    sections.append("**Research used:**\n" + ("\n".join(hooks) or "- Nothing specific found; the message is based on the role and team."))
    flagged = result.get("feedback", {}).get(name)
    if flagged:
        sections.append("**⚠️ The checker still flagged:**\n" + "\n".join(f"- {f}" for f in flagged))
    return note, d.get("email_subject", ""), d.get("email_body", ""), "\n\n".join(sections)


GMAIL_COMPOSE = "https://mail.google.com/mail/"


def gmail_compose_url(to: str, subject: str, body: str, account: str = "") -> str:
    """A Gmail link that opens a filled-in compose window. Gmail never sends from a link: you press Send."""
    params = {"view": "cm", "fs": "1", "to": to, "su": subject, "body": body}
    if account:
        params["authuser"] = account          # which signed-in Google account to open
    return GMAIL_COMPOSE + "?" + urlencode(params, quote_via=quote)


def compose_button(result: dict, name: str | None, subject: str, body: str, account: str = "") -> str:
    """HTML for the 'Open in Gmail' button, or the reason it isn't available.
    A draft the checker still flags stays locked until its subject or body has been edited."""
    if not name:
        return ""
    person = next((p for p in result.get("selected", []) if p["name"] == name), {})
    if not person.get("email"):
        return f"<p>No email address found for {html.escape(name)}. Copy the message instead.</p>"
    draft = result.get("drafts", {}).get(name, {})
    flagged = result.get("feedback", {}).get(name)
    edited = (subject, body) != (draft.get("email_subject", ""), draft.get("email_body", ""))
    if flagged and not edited:
        return ("<p><b>Gmail is locked for this draft:</b> the checker still flags it (see the right column). "
                "Edit the subject or email to fix those points and the button appears.</p>")
    notes = []
    if flagged:
        notes.append("You edited a flagged draft. Reread it before sending.")
    if person.get("email_status") != "valid":
        notes.append(f"This address is {html.escape(person.get('email_status') or 'unverified')}, so it may bounce.")
    url = html.escape(gmail_compose_url(person["email"], subject, body, account), quote=True)
    link = (f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            'style="display:inline-block;padding:10px 18px;border-radius:8px;background:#ea4335;color:#fff;'
            'font-weight:600;text-decoration:none">Open in Gmail (opens a draft, does not send)</a>')
    return link + "".join(f"<p>{n}</p>" for n in notes)


def tracker_rows(settings: Settings) -> list[list[str]]:
    path = settings.outputs / "tracker.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.reader(f))[1:][-50:][::-1]           # newest first


def build_app(runner: OutreachRunner):
    import gradio as gr

    from .report import TRACKER_FIELDS, rewrite_report

    async def on_find(company, role, job_url, domain):
        try:
            out = await runner.start(company, role, job_url, domain)
        except AgentError as e:
            raise gr.Error(str(e))
        return show_shortlist(out)

    async def on_load(run_id):
        try:
            out = await runner.load_paused(run_id)
        except AgentError as e:
            raise gr.Error(str(e))
        return show_shortlist(out)

    def show_shortlist(out):
        choices = [contact_label(i, p) for i, p in enumerate(out["shortlist"])]
        status = "\n".join(out["logs"])
        if not choices:
            status += "\n\nNo suitable recruiters or engineering managers found. Try the email domain field."
        else:
            status += f"\n\nPaused for your review. Run id: {out['run_id']}"
        return (status, gr.CheckboxGroup(choices=choices, value=choices, visible=bool(choices)),
                gr.Button(visible=bool(choices)), {"run_id": out["run_id"], "choices": choices})

    async def on_draft(selected, session):
        if not session or not session.get("run_id"):
            raise gr.Error("Find people first.")
        indices = [session["choices"].index(s) for s in selected if s in session["choices"]]
        try:
            out = await runner.approve(session["run_id"], indices)
        except AgentError as e:
            raise gr.Error(str(e))
        result = out["result"]
        names = [p["name"] for p in result.get("selected", [])]
        status = "\n".join(out["logs"]) or "Nothing selected, nothing drafted."
        if result.get("metrics") and names:
            status += "\n\n" + one_line(summarize(result["metrics"], runner.settings))
        first = names[0] if names else None
        fields = draft_fields(result, first) if first else ("", "", "", "")
        gmail = compose_button(result, first, fields[1], fields[2], runner.settings.gmail_account)
        # session["result"] keeps the agent's drafts untouched; your changes live in session["edits"].
        return (status, gr.Column(visible=bool(names)), gr.Dropdown(choices=names, value=first), *fields,
                note_counter(fields[0]), gmail, result.get("report_path"), tracker_rows(runner.settings),
                {**session, "result": result, "edits": {}}, "")

    def on_pick(name, session):
        session = session or {}
        result, edits = session.get("result", {}), session.get("edits", {})
        fields = draft_fields(result, name, edits) if name else ("", "", "", "")
        return (*fields, note_counter(fields[0]),
                compose_button(result, name, fields[1], fields[2], runner.settings.gmail_account))

    def on_edit(name, note_text, subject_text, body_text, session):
        session = dict(session or {})
        result = session.get("result", {})
        if name:
            edits = dict(session.get("edits", {}))
            draft = {k: result.get("drafts", {}).get(name, {}).get(k, "") for k in EDIT_KEYS}
            typed = {"linkedin_note": note_text, "email_subject": subject_text, "email_body": body_text}
            if typed == draft:
                edits.pop(name, None)
            else:
                edits[name] = typed
            session["edits"] = edits
        gmail = compose_button(result, name, subject_text, body_text, runner.settings.gmail_account)
        return note_counter(note_text), gmail, session

    def on_reset(name, session):
        session = dict(session or {})
        session["edits"] = {k: v for k, v in session.get("edits", {}).items() if k != name}
        return (*on_pick(name, session), session)

    def on_save(session):
        session = session or {}
        result, edits = session.get("result", {}), session.get("edits", {})
        if not result.get("report_path"):
            raise gr.Error("Nothing to save yet.")
        path = rewrite_report(apply_edits(result, edits), Path(result["report_path"]))
        who = ", ".join(sorted(edits)) or "no one (nothing was changed)"
        return str(path), f"Saved. Edited drafts in the report: {who}."

    with gr.Blocks(title="Outreach Agent") as app:
        gr.Markdown("# Outreach Agent\nFinds the right people, researches them, and drafts fact-checked "
                    "messages from your resume. **It never sends anything:** you edit and send each one yourself.")
        session = gr.State({})

        with gr.Tab("Draft outreach"):
            with gr.Row():
                company = gr.Textbox(label="Company", placeholder="Stripe")
                role = gr.Textbox(label="Role", placeholder="AI Engineer, New Grad")
            with gr.Row():
                job_url = gr.Textbox(label="Job posting URL (optional, makes messages more specific)")
                domain = gr.Textbox(label="Email domain (optional)", placeholder="stripe.com")
            find = gr.Button("1. Find people", variant="primary")
            with gr.Accordion("Resume a paused run", open=False):
                run_id = gr.Textbox(label="Run id")
                load = gr.Button("Load")
            status = gr.Textbox(label="Progress", lines=6, max_lines=14, interactive=False)
            picks = gr.CheckboxGroup(label="Who should I research and draft for?", visible=False)
            draft = gr.Button("2. Research and draft for the selected people", variant="primary", visible=False)

            with gr.Column(visible=False) as drafts_box:
                contact = gr.Dropdown(label="Drafts")
                gr.Markdown("**Edit freely.** Your changes are kept when you switch between people, go into the "
                            "Gmail draft and the copy buttons, and are written to the report when you save.")
                with gr.Row():
                    with gr.Column(scale=3):
                        note = gr.Textbox(label="LinkedIn note", lines=3, interactive=True, buttons=["copy"])
                        counter = gr.Markdown()
                        subject = gr.Textbox(label="Email subject", interactive=True, buttons=["copy"])
                        body = gr.Textbox(label="Email", lines=10, interactive=True, buttons=["copy"])
                        gmail = gr.HTML()
                        with gr.Row():
                            save = gr.Button("Save my edits to the report")
                            reset = gr.Button("Undo my edits for this person", variant="secondary")
                        saved = gr.Markdown()
                    with gr.Column(scale=2):
                        notes = gr.Markdown()
                report = gr.File(label="Full report (Markdown)")

        with gr.Tab("Tracker"):
            tracker = gr.Dataframe(value=tracker_rows(runner.settings), headers=TRACKER_FIELDS,
                                   interactive=False, wrap=True)
            gr.Markdown("Update the `status` column in `outputs/tracker.csv` as replies come in.")

        find.click(on_find, [company, role, job_url, domain], [status, picks, draft, session])
        load.click(on_load, [run_id], [status, picks, draft, session])
        draft.click(on_draft, [picks, session], [status, drafts_box, contact, note, subject, body, notes,
                                                 counter, gmail, report, tracker, session, saved])
        contact.change(on_pick, [contact, session], [note, subject, body, notes, counter, gmail])
        for box in (note, subject, body):      # .input fires only when you type, not when we fill the boxes
            box.input(on_edit, [contact, note, subject, body, session], [counter, gmail, session])
        reset.click(on_reset, [contact, session], [note, subject, body, notes, counter, gmail, session])
        save.click(on_save, [session], [report, saved])
    return app


def main() -> None:
    build_app(OutreachRunner()).launch()


if __name__ == "__main__":
    main()
