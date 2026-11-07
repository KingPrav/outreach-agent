import csv

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from outreach_agent.clients import AgentError
from outreach_agent.config import Settings
from outreach_agent.graph import build_graph

from .fakes import FakeLLM, people_tools, research_tools

CONFIG = {"configurable": {"thread_id": "t1"}}
INPUTS = {"company": "Northwind", "role": "AI Engineer", "domain": "", "job_url": "https://northwind.dev/jobs/1"}


@pytest.fixture
def settings(tmp_path):
    resume = tmp_path / "resume.md"
    resume.write_text("Alex Rivera. MS CS. Built a RAG eval harness.")
    return Settings(resume_path=resume, notes_path=tmp_path / "notes.md", max_people=6, outputs=tmp_path / "out")


def graph(settings, llm, people=None, research=None):
    return build_graph(people or people_tools(), research or research_tools(), llm, settings,
                       checkpointer=InMemorySaver(), log=lambda *_: None)


async def run_until_review(app):
    result = await app.ainvoke(INPUTS, CONFIG)
    assert "__interrupt__" in result
    return result["__interrupt__"][0].value


async def test_full_run_pauses_for_review_then_drafts_and_saves(settings):
    llm = FakeLLM()
    app = graph(settings, llm)

    info = await run_until_review(app)
    assert [p["name"] for p in info["shortlist"]] == ["Sam Patel", "Jordan Lee"]   # sales contact skipped
    assert llm.calls["Draft"] == 0                                                    # nothing drafted before approval

    result = await app.ainvoke(Command(resume=[0, 1]), CONFIG)
    assert llm.calls["Draft"] == 2 and llm.calls["FactCheck"] == 2
    assert result["feedback"] == {}
    assert all(p["email_status"] == "valid" for p in result["selected"])

    report = (settings.outputs / result["report_path"].split("/")[-1]).read_text()
    assert "Jordan Lee" in report and "Sam Patel" in report
    rows = list(csv.DictReader((settings.outputs / "tracker.csv").open()))
    assert [r["name"] for r in rows] == ["Sam Patel", "Jordan Lee"]


async def test_hooks_must_cite_a_retrieved_url(settings):
    app = graph(settings, FakeLLM())
    await run_until_review(app)
    result = await app.ainvoke(Command(resume=[0]), CONFIG)
    hooks = result["research"][0]["hooks"]
    assert [h["fact"] for h in hooks] == ["Spoke about eval tooling"]   # the invented URL was dropped


async def test_no_hooks_when_model_is_unsure_it_is_the_same_person(settings):
    app = graph(settings, FakeLLM(unconfident={"Sam Patel"}))
    await run_until_review(app)
    result = await app.ainvoke(Command(resume=[0]), CONFIG)
    assert result["research"][0]["hooks"] == []


async def test_failed_draft_is_redrafted_once_and_only_that_one(settings):
    llm = FakeLLM(bad_first_draft={"Sam Patel"})
    app = graph(settings, llm)
    await run_until_review(app)
    result = await app.ainvoke(Command(resume=[0, 1]), CONFIG)
    assert llm.drafted == {"Jordan Lee": 1, "Sam Patel": 2}
    assert result["revisions"] == 2
    assert result["feedback"] == {}
    assert "[Company]" not in result["drafts"]["Sam Patel"]["email_body"]


async def test_selecting_nobody_ends_without_drafting(settings):
    llm = FakeLLM()
    app = graph(settings, llm)
    await run_until_review(app)
    result = await app.ainvoke(Command(resume=[]), CONFIG)
    assert "report_path" not in result and llm.calls["Draft"] == 0


async def test_empty_shortlist_finishes_without_pausing(settings):
    app = graph(settings, FakeLLM(picks=[]))
    result = await app.ainvoke(INPUTS, CONFIG)
    assert "__interrupt__" not in result and result["shortlist"] == []


async def test_company_lookup_error_falls_back_to_domains(settings):
    people = people_tools(error_for_company="Hunter API error 400: company not found")
    app = graph(settings, FakeLLM(), people=people)
    info = await run_until_review(app)
    assert info["shortlist"]
    domains_tried = [a.get("domain") for name, a in people.calls if name == "find_people" and "domain" in a]
    assert "northwind.dev" in domains_tried


async def test_out_of_credits_stops_with_a_clear_error(settings):
    people = people_tools(error_for_company="Hunter API error 429: usage limit reached")
    app = graph(settings, FakeLLM(), people=people)
    with pytest.raises(AgentError, match="429"):
        await app.ainvoke(INPUTS, CONFIG)


async def test_missing_resume_is_a_clear_error(settings, tmp_path):
    missing = Settings(resume_path=tmp_path / "nope.pdf", notes_path=settings.notes_path, outputs=settings.outputs)
    app = graph(missing, FakeLLM())
    with pytest.raises(AgentError, match="Resume not found"):
        await app.ainvoke(INPUTS, CONFIG)


async def test_profile_is_cached_between_runs(settings):
    llm = FakeLLM()
    await graph(settings, llm).ainvoke(INPUTS, CONFIG)
    await graph(settings, llm).ainvoke(INPUTS, {"configurable": {"thread_id": "t2"}})
    assert llm.calls["CandidateProfile"] == 1


async def test_paused_review_survives_a_restart(settings, tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db = str(tmp_path / "checkpoints.sqlite")
    quiet = dict(log=lambda *_: None)
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        first = build_graph(people_tools(), research_tools(), FakeLLM(), settings, checkpointer=saver, **quiet)
        await first.ainvoke(INPUTS, CONFIG)                       # pauses at review

    async with AsyncSqliteSaver.from_conn_string(db) as saver:    # a new process, same database
        second = build_graph(people_tools(), research_tools(), FakeLLM(), settings, checkpointer=saver, **quiet)
        snapshot = await second.aget_state(CONFIG)
        assert snapshot.tasks[0].interrupts                        # still waiting for the review
        result = await second.ainvoke(Command(resume=[0]), CONFIG)
    assert result["report_path"]
