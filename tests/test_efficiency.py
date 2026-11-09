"""Retries, run stats, parallel ordering and prompt layout."""

import json
import re
from types import SimpleNamespace

from langgraph.types import Command

from outreach_agent.clients import TRANSIENT, TRANSIENT_NO_429, MCPTools
from outreach_agent.config import Settings
from outreach_agent.metrics import summarize, to_markdown

from .fakes import FakeLLM, people_tools, research_tools
from .test_graph import CONFIG, INPUTS, graph, settings  # noqa: F401  (fixture)


class FlakySession:
    """Returns the queued errors first, then a good answer."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    async def call_tool(self, name, args):
        self.calls += 1
        text = json.dumps({"error": self.errors.pop(0)} if self.errors else {"ok": True})
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=text)])


async def no_sleep(_):
    return None


async def test_transient_errors_are_retried_then_succeed():
    session = FlakySession("Tavily API error 503: unavailable", "Read timed out")
    tools = MCPTools(session, retry_on=TRANSIENT, sleep=no_sleep)
    assert await tools.call("web_search", {}) == {"ok": True}
    assert session.calls == 3


async def test_retries_stop_after_the_last_attempt():
    session = FlakySession(*["Tavily API error 502"] * 5)
    out = await MCPTools(session, retry_on=TRANSIENT, attempts=3, sleep=no_sleep).call("web_search", {})
    assert "502" in out["error"] and session.calls == 3


async def test_hunter_quota_and_bad_requests_are_not_retried():
    for error in ("Hunter API error 429: usage limit reached", "Hunter API error 400: bad domain"):
        session = FlakySession(error)
        out = await MCPTools(session, retry_on=TRANSIENT_NO_429, sleep=no_sleep).call("find_people", {})
        assert out["error"] == error and session.calls == 1


async def test_domain_search_only_runs_when_the_name_lookup_fails(settings):  # noqa: F811
    research = research_tools()
    await graph(settings, FakeLLM(), research=research).ainvoke(INPUTS, CONFIG)
    assert not any("official website" in a["query"] for n, a in research.calls if n == "web_search")

    research = research_tools()
    people = people_tools(error_for_company="Hunter API error 400: company not found")
    await graph(settings, FakeLLM(), people=people, research=research).ainvoke(
        INPUTS, {"configurable": {"thread_id": "fallback"}})
    assert any("official website" in a["query"] for n, a in research.calls if n == "web_search")


async def test_role_research_is_done_before_review_and_verify_overlaps_research(settings):  # noqa: F811
    llm = FakeLLM()
    app = graph(settings, llm)
    await app.ainvoke(INPUTS, CONFIG)
    assert llm.calls["RoleBrief"] == 1 and llm.calls["Draft"] == 0      # role brief ready at the pause
    result = await app.ainvoke(Command(resume=[0, 1]), CONFIG)
    rows = {m["node"]: m for m in result["metrics"]}
    assert rows["verify_emails"]["start"] < rows["draft"]["start"]
    assert llm.calls["RoleBrief"] == 1                                   # not repeated after the review


async def test_draft_and_check_prompts_put_shared_context_first(settings):  # noqa: F811
    llm = FakeLLM()
    app = graph(settings, llm)
    await app.ainvoke(INPUTS, CONFIG)
    await app.ainvoke(Command(resume=[0, 1]), CONFIG)
    for schema in ("Draft", "FactCheck"):
        users = [m[-1]["content"] for m in llm.messages[schema]]
        assert all(list(json.loads(u))[0] == "candidate_profile" for u in users)
        prefixes = {u[:u.index('"recipient"' if schema == "Draft" else '"person_research"')] for u in users}
        assert len(prefixes) == 1          # byte-identical shared prefix across people


async def test_run_stats_reach_the_report(settings):  # noqa: F811
    from dataclasses import replace

    priced = replace(settings, price_input=0.15, price_cached_input=0.075, price_output=0.6)
    llm = FakeLLM()
    app = graph(priced, llm)
    await app.ainvoke(INPUTS, CONFIG)
    result = await app.ainvoke(Command(resume=[0, 1]), CONFIG)

    stats = summarize(result["metrics"], priced)
    assert stats["totals"]["llm_calls"] == sum(llm.calls.values())
    assert stats["totals"]["cached_input_tokens"] == 40 * sum(llm.calls.values())
    assert stats["cost"] > 0
    report = (priced.outputs / result["report_path"].split("/")[-1]).read_text()
    assert "## Run stats" in report and "| draft |" in report and "Estimated model cost" in report


def test_stats_summary_and_markdown():
    rows = [{"node": "draft", "start": 10.0, "seconds": 2.0, "llm_calls": 2, "input_tokens": 1000,
             "cached_input_tokens": 600, "output_tokens": 200},
            {"node": "draft", "start": 13.0, "seconds": 1.0, "llm_calls": 1, "input_tokens": 500,
             "cached_input_tokens": 0, "output_tokens": 100},
            {"node": "shortlist", "start": 1.0, "seconds": 1.0, "llm_calls": 1, "input_tokens": 100,
             "cached_input_tokens": 0, "output_tokens": 10}]
    s = summarize(rows, Settings(price_input=None, price_output=None))
    draft = next(n for n in s["nodes"] if n["node"] == "draft")
    assert draft["runs"] == 2 and draft["llm_calls"] == 3 and s["wait_after_review"] == 4.0
    assert s["cost"] is None
    md = to_markdown(s)
    assert "38% served from OpenAI's prompt cache" in md and "Estimated model cost" not in md
    assert re.search(r"\| draft \| 2 \| 3\.00 \|", md)


def test_saving_edits_keeps_run_stats(tmp_path):
    from outreach_agent.report import rewrite_report

    path = tmp_path / "r.md"
    path.write_text("# Old\n\n## Run stats\n\n- Wait after you approved contacts: **3.0 s**\n")
    state = {"company": "N", "role": "R", "research": [], "drafts": {}, "selected": []}
    text = rewrite_report(state, path).read_text()
    assert text.startswith("# Outreach: R at N") and "## Run stats" in text and "# Old" not in text
