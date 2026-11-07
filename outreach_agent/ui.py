"""Web UI (Gradio) for the outreach agent.

    pip install -e ".[ui]"
    python -m outreach_agent.ui          # opens http://127.0.0.1:7860

The UI drives the same graph as the command line. Each button opens the MCP servers, runs the
graph until it pauses or finishes, and closes them; the run's state lives in the SQLite
checkpointer between clicks, keyed by the run id. So a review can be resumed later, from the UI
or the command line.
"""

import csv
from contextlib import AbstractAsyncContextManager
from typing import Callable

from langgraph.types import Command

from .cli import new_run_id
from .clients import AgentError
from .config import LINKEDIN_NOTE_LIMIT, Settings
from .graph import build_graph
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


def draft_fields(result: dict, name: str) -> tuple[str, str, str, str]:
    """(LinkedIn note, subject, email body, notes markdown) for one contact."""
    d = result.get("drafts", {}).get(name, {})
    research = next((r for r in result.get("research", []) if r["name"] == name), {})
    person = next((p for p in result.get("selected", []) if p["name"] == name), {})
    note = d.get("linkedin_note", "")
    sections = [f"**Email:** {person.get('email') or 'not found'} ({person.get('email_status') or 'unverified'})"]
    if person.get("linkedin"):
        sections.append(f"**LinkedIn:** {person['linkedin']}")
    sections.append(f"**LinkedIn note length:** {len(note)}/{LINKEDIN_NOTE_LIMIT} characters")
    hooks = [f"- {h['fact']} ([source]({h['source_url']}))" for h in research.get("hooks", [])]
    sections.append("**Research used:**\n" + ("\n".join(hooks) or "- Nothing specific found; the message is based on the role and team."))
    flagged = result.get("feedback", {}).get(name)
    if flagged:
        sections.append("**⚠️ The checker still flagged:**\n" + "\n".join(f"- {f}" for f in flagged))
    return note, d.get("email_subject", ""), d.get("email_body", ""), "\n\n".join(sections)


def tracker_rows(settings: Settings) -> list[list[str]]:
    path = settings.outputs / "tracker.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.reader(f))[1:][-50:][::-1]           # newest first


def build_app(runner: OutreachRunner):
    import gradio as gr

    from .report import TRACKER_FIELDS

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
        first = names[0] if names else None
        fields = draft_fields(result, first) if first else ("", "", "", "")
        report = result.get("report_path")
        return (status, gr.Column(visible=bool(names)), gr.Dropdown(choices=names, value=first), *fields,
                report, tracker_rows(runner.settings), {**session, "result": result})

    def on_pick(name, session):
        return draft_fields((session or {}).get("result", {}), name) if name else ("", "", "", "")

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
                with gr.Row():
                    with gr.Column(scale=3):
                        note = gr.Textbox(label="LinkedIn note", lines=3, buttons=["copy"])
                        subject = gr.Textbox(label="Email subject", buttons=["copy"])
                        body = gr.Textbox(label="Email", lines=10, buttons=["copy"])
                    with gr.Column(scale=2):
                        notes = gr.Markdown()
                gr.Markdown("Edits here are for copying; they are not saved back to the report.")
                report = gr.File(label="Full report (Markdown)")

        with gr.Tab("Tracker"):
            tracker = gr.Dataframe(value=tracker_rows(runner.settings), headers=TRACKER_FIELDS,
                                   interactive=False, wrap=True)
            gr.Markdown("Update the `status` column in `outputs/tracker.csv` as replies come in.")

        find.click(on_find, [company, role, job_url, domain], [status, picks, draft, session])
        load.click(on_load, [run_id], [status, picks, draft, session])
        draft.click(on_draft, [picks, session],
                    [status, drafts_box, contact, note, subject, body, notes, report, tracker, session])
        contact.change(on_pick, [contact, session], [note, subject, body, notes])
    return app


def main() -> None:
    build_app(OutreachRunner()).launch()


if __name__ == "__main__":
    main()
