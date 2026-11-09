from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from outreach_agent.clients import AgentError
from outreach_agent.runtime import Services
from outreach_agent.ui import OutreachRunner, contact_label, draft_fields

from .fakes import FakeLLM, people_tools, research_tools
from .test_graph import settings  # noqa: F401  (fixture)


@pytest.fixture
def runner(settings):  # noqa: F811
    saver, llm = InMemorySaver(), FakeLLM()   # one checkpointer for every click, like the SQLite file

    @asynccontextmanager
    async def fake_services(_settings):
        yield Services(people_tools(), research_tools(), llm, saver)

    return OutreachRunner(settings, fake_services)


async def test_find_then_draft(runner):
    found = await runner.start("Northwind", "AI Engineer", "https://northwind.dev/jobs/1")
    assert [p["name"] for p in found["shortlist"]] == ["Sam Patel", "Jordan Lee"]
    assert any("[shortlist]" in line for line in found["logs"])

    done = (await runner.approve(found["run_id"], [1]))["result"]
    assert [p["name"] for p in done["selected"]] == ["Jordan Lee"]
    note, subject, body, notes = draft_fields(done, "Jordan Lee")
    assert "Jordan" in note and "Hi Jordan" in body and subject
    assert "Spoke about eval tooling" in notes and "Invented fact" not in notes


async def test_paused_run_can_be_reloaded(runner):
    found = await runner.start("Northwind", "AI Engineer")
    again = await runner.load_paused(found["run_id"])
    assert again["shortlist"] == found["shortlist"]


async def test_unknown_run_id_is_a_clear_error(runner):
    with pytest.raises(AgentError, match="No paused run"):
        await runner.load_paused("does-not-exist")


async def test_company_and_role_are_required(runner):
    with pytest.raises(AgentError):
        await runner.start("", "AI Engineer")


def test_contact_label():
    p = {"name": "Jordan Lee", "position": "EM", "category": "engineering_manager", "why": "ML platform"}
    assert contact_label(0, p) == "1. Jordan Lee, EM (engineering manager): ML platform"


def test_app_builds(runner):
    pytest.importorskip("gradio")
    from outreach_agent.ui import build_app

    app = build_app(runner)
    assert app.title == "Outreach Agent"


def test_gmail_url_encodes_everything():
    from urllib.parse import parse_qs, urlparse

    from outreach_agent.ui import gmail_compose_url

    url = gmail_compose_url("a@b.dev", "Hi & welcome", "Line one\n\nLine two: 100% sure?", "me@gmail.com")
    q = parse_qs(urlparse(url).query)
    assert url.startswith("https://mail.google.com/mail/?")
    assert q == {"view": ["cm"], "fs": ["1"], "to": ["a@b.dev"], "su": ["Hi & welcome"],
                 "body": ["Line one\n\nLine two: 100% sure?"], "authuser": ["me@gmail.com"]}


RESULT = {
    "selected": [{"name": "Jordan Lee", "email": "jordan@northwind.dev", "email_status": "valid"},
                 {"name": "Sam Patel", "email": None},
                 {"name": "Casey Morgan", "email": "casey@northwind.dev", "email_status": "accept_all"}],
    "drafts": {"Jordan Lee": {"email_subject": "Hello", "email_body": "Body"},
               "Casey Morgan": {"email_subject": "Hi", "email_body": "Text"}},
    "feedback": {"Jordan Lee": ["Embellished claim"]},
}


def test_gmail_button_locked_while_flagged_draft_is_unedited():
    from outreach_agent.ui import compose_button

    assert "locked" in compose_button(RESULT, "Jordan Lee", "Hello", "Body")
    edited = compose_button(RESULT, "Jordan Lee", "Hello", "Body, fixed")
    assert "mail.google.com" in edited and "Reread it" in edited and 'target="_blank"' in edited


def test_gmail_button_without_email_or_with_unverified_email():
    from outreach_agent.ui import compose_button

    assert "No email address" in compose_button(RESULT, "Sam Patel", "", "")
    unverified = compose_button(RESULT, "Casey Morgan", "Hi", "Text")
    assert "mail.google.com" in unverified and "accept_all" in unverified
    assert compose_button(RESULT, None, "", "") == ""


def test_edits_override_the_draft_and_are_marked_when_saved(tmp_path):
    from outreach_agent.report import rewrite_report
    from outreach_agent.ui import apply_edits, current_text, draft_fields, note_counter

    result = {"company": "Northwind", "role": "AI Engineer", "research": [],
              "selected": [{"name": "Jordan Lee", "position": "EM", "category": "engineering_manager"},
                           {"name": "Sam Patel", "position": "Recruiter", "category": "recruiter"}],
              "drafts": {"Jordan Lee": {"linkedin_note": "AI note", "email_subject": "AI subject", "email_body": "AI body"},
                         "Sam Patel": {"linkedin_note": "Sam note", "email_subject": "S", "email_body": "Sam body"}}}
    edits = {"Jordan Lee": {"linkedin_note": "My note", "email_subject": "AI subject", "email_body": "My body"}}

    assert current_text(result, edits, "Jordan Lee")["email_body"] == "My body"
    assert current_text(result, edits, "Sam Patel")["email_body"] == "Sam body"
    assert draft_fields(result, "Jordan Lee", edits)[0] == "My note"

    saved = apply_edits(result, edits)
    assert saved["drafts"]["Jordan Lee"]["edited"] and "edited" not in saved["drafts"]["Sam Patel"]
    assert result["drafts"]["Jordan Lee"]["email_body"] == "AI body"      # the agent's draft is untouched

    path = rewrite_report(saved, tmp_path / "report.md")
    text = path.read_text()
    assert "My body" in text and "AI body" not in text and text.count("_Edited by you._") == 1
    assert not (tmp_path / "tracker.csv").exists()                         # saving never adds tracker rows

    assert note_counter("x" * 10) == "10/300 characters"
    assert "cut it off" in note_counter("x" * 301)


async def test_status_shows_run_stats_after_drafting(runner):
    from outreach_agent.metrics import one_line, summarize

    found = await runner.start("Northwind", "AI Engineer")
    result = (await runner.approve(found["run_id"], [0]))["result"]
    line = one_line(summarize(result["metrics"], runner.settings))
    assert line.startswith("Wait after approval:") and "model calls" in line
