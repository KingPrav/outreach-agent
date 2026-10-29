"""
agent.py: a LangGraph agent that prepares personalized cold outreach for a job.

Give it a company and a role. It finds recruiters and engineering managers,
lets you pick who to contact, researches each person and the role, and drafts a
LinkedIn note and an email for each one, grounded in your resume.
It never sends anything: you review, edit and send the messages yourself.

    load_profile -> find_domain -> find_people -> shortlist -> [YOU REVIEW]
        -> research_person (one per person, in parallel) + research_role
        -> draft -> check --(problems? redraft once)--> verify_emails -> save

Run it:
    python agent.py --company "Stripe" --role "Backend Engineer"
    python agent.py --company "Stripe" --role "Backend Engineer" --domain stripe.com --job-url https://...
    python agent.py --graph        # print the graph as a Mermaid diagram
"""

import argparse
import asyncio
import csv
import hashlib
import json
import operator
import os
import re
import sys
from contextlib import AsyncExitStack
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RESUME_PATH = ROOT / os.getenv("RESUME_PATH", "profile/resume.pdf")
NOTES_PATH = ROOT / "profile" / "notes.md"     # optional: your extra instructions
MAX_PEOPLE = int(os.getenv("MAX_PEOPLE", "6"))
OUTPUTS = ROOT / "outputs"
LINKEDIN_NOTE_LIMIT = 300                       # LinkedIn's connection-note limit
EMAIL_WORD_LIMIT = 160


# ===========================================================================
# 1. STATE: shared data every node reads and writes
# ===========================================================================
class State(TypedDict, total=False):
    company: str
    role: str
    domain: str
    job_url: str
    profile: dict                                       # your resume, structured
    candidates: list[dict]                              # everyone Hunter found
    shortlist: list[dict]                               # who the model suggests
    selected: list[dict]                                # who YOU approved
    role_brief: dict                                    # what we learned about the role
    research: Annotated[list[dict], operator.add]       # one entry per person (parallel nodes append)
    drafts: dict                                        # name -> draft
    feedback: dict                                      # name -> problems found by check
    revisions: int
    log: Annotated[list[str], operator.add]             # progress notes


# ===========================================================================
# 2. STRUCTURED OUTPUT SCHEMAS: the exact shapes we ask the model to return
# ===========================================================================
class Highlight(BaseModel):
    title: str = Field(description="Project or job, e.g. 'HireRank: Resume Matching Service'")
    detail: str = Field(description="One sentence on what was built or done, using only facts from the resume")
    metrics: list[str] = Field(description="Exact numbers from the resume for this item, if any")


class CandidateProfile(BaseModel):
    name: str
    headline: str = Field(description="e.g. 'MS CS student at Boston University (Dec 2026), AI engineer'")
    education: list[str]
    skills: list[str]
    highlights: list[Highlight]


class Pick(BaseModel):
    index: int = Field(description="The candidate's number in the list")
    category: Literal["recruiter", "engineering_manager", "skip"]
    reason: str = Field(description="One short reason")


class Shortlist(BaseModel):
    picks: list[Pick]


class Hook(BaseModel):
    fact: str = Field(description="One specific, professional fact about the person, e.g. a talk or post")
    source_url: str = Field(description="The URL the fact came from (must be one of the search results)")


class PersonResearch(BaseModel):
    same_person_confident: bool = Field(description="True only if the results clearly refer to this person at this company")
    hooks: list[Hook] = Field(description="Up to 3 hooks. Empty list if nothing specific and reliable was found.")
    summary: str = Field(description="One sentence on what this person works on, or '' if unknown")


class RoleBrief(BaseModel):
    summary: str = Field(description="2-3 sentences on the role and team")
    key_requirements: list[str]
    talking_points: list[str] = Field(description="Specific things about the company/team worth mentioning, each with its source URL in brackets")


class Draft(BaseModel):
    linkedin_note: str = Field(description=f"Connection note, at most {LINKEDIN_NOTE_LIMIT - 20} characters")
    email_subject: str
    email_body: str = Field(description=f"Plain-text email, at most {EMAIL_WORD_LIMIT} words, signed with the candidate's name")
    hook_used: str = Field(description="Which fact about the person or company the message is built on, or 'none'")
    resume_points_used: list[str] = Field(description="Which resume highlights the message mentions")


class FactCheck(BaseModel):
    problems: list[str] = Field(description="Claims not supported by the profile, research or role brief; generic filler; wrong names. Empty if fine.")


# ===========================================================================
# 3. PROMPTS
# ===========================================================================
PROFILE_PROMPT = """Extract a structured profile from this resume. Use ONLY what the resume says:
no invented skills, numbers or achievements. Keep each metric exactly as written.
The resume text may have bullets separated from their headings; match each bullet to the
project or job it most clearly describes."""

SHORTLIST_PROMPT = """You help a job seeker choose who to contact at {company} about a "{role}" role.
For each candidate, choose a category:
- "recruiter": recruiters, technical recruiters, talent acquisition, university/early-career recruiting.
- "engineering_manager": engineering managers, heads/directors/VPs of engineering, tech leads who manage a team
  relevant to the role.
- "skip": everyone else (individual engineers with no hiring role, sales, finance, legal, etc.).
Prefer people whose title matches the role's area (e.g. ML roles -> ML/AI engineering leaders).
Return at most {max_people} non-skip picks, best first."""

PERSON_PROMPT = """From these web search results, find up to 3 specific, PROFESSIONAL hooks about {name}
({position} at {company}) that a job seeker could mention in a short message: talks, blog posts,
podcasts, open-source work, launches they led, interviews.
Rules:
- Only use results that are clearly about THIS person at THIS company. Same-name strangers are common.
- Nothing personal (family, health, location, politics, hobbies). Professional work only.
- Every hook must cite a URL from the results. If nothing reliable is found, return no hooks."""

ROLE_PROMPT = """Summarize what a candidate should know about the "{role}" role at {company}, from the
material below (a job posting and/or search results). Only use facts in the material, and put the
source URL in [brackets] after each talking point."""

DRAFT_PROMPT = """Write cold outreach from a job seeker to one person. Return a LinkedIn connection note
and an email.

Rules:
- Use ONLY facts from the candidate profile, the person research and the role brief. Never invent
  experience, numbers, mutual connections or familiarity ("I've followed your work for years").
- Open with ONE specific hook about the person or their team (from the research). If there are no
  hooks, use a specific fact about the team or role instead. Never pretend to know them.
- Connect it to ONE or TWO relevant points from the candidate's resume, matched to the role.
- End with ONE small, clear ask:
    recruiter -> whether they're the right person for the {role} role, or a short chat about fit;
    engineering manager -> a 15-minute chat about the team's work, or advice on applying.
- Sound like a real person: specific, warm, concise. No "I hope this email finds you well",
  no buzzwords, no flattery, no exclamation marks in every sentence.
- LinkedIn note: at most {note_limit} characters. Email: at most {word_limit} words, signed with the
  candidate's name.
{extra}"""

CHECK_PROMPT = """You are a strict reviewer of a cold outreach draft. List every problem:
- Any claim about the candidate that is NOT in the candidate profile (skills, numbers, experience).
- Any claim about the recipient or company that is NOT in the research or role brief.
- Pretended familiarity, generic filler, or the wrong name/company.
Return an empty list if the draft is accurate and specific."""


# ===========================================================================
# Helpers
# ===========================================================================
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def read_resume(path: Path) -> str:
    if not path.exists():
        sys.exit(f"Resume not found at {path}. Put it in profile/ and set RESUME_PATH in .env.")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text()


def tool_json(result) -> dict:
    """MCP tool results come back as text; our servers always return JSON."""
    return json.loads(result.content[0].text)


NOT_COMPANY_SITES = (
    "wikipedia.org", "linkedin.com", "crunchbase.com", "glassdoor.com", "indeed.com", "bloomberg.com",
    "forbes.com", "youtube.com", "x.com", "twitter.com", "facebook.com", "instagram.com", "github.com",
    "gitlab.com", "medium.com", "reddit.com", "builtin.com", "wellfound.com", "zoominfo.com", "pitchbook.com",
    "levels.fyi", "yahoo.com", "reuters.com", "cnbc.com", "techcrunch.com",
    # job boards / applicant tracking systems: a job link on these doesn't tell us the company's domain
    "greenhouse.io", "lever.co", "ashbyhq.com", "myworkdayjobs.com", "workday.com", "smartrecruiters.com",
    "jobvite.com", "icims.com", "bamboohr.com", "workable.com",
)


def base_domain(url_or_host: str) -> str:
    """'https://careers.datadoghq.com/x' -> 'datadoghq.com' (drops subdomains like careers., www.)."""
    host = urlparse(url_or_host if "//" in url_or_host else f"//{url_or_host}").netloc.lower().split(":")[0]
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])        # e.g. company.co.uk
    return ".".join(parts[-2:])


def is_company_site(domain: str) -> bool:
    return bool(domain) and not any(domain == s or domain.endswith("." + s) for s in NOT_COMPANY_SITES)


# ===========================================================================
# 4. NODES
# ===========================================================================
def build_graph(people: ClientSession, research: ClientSession, llm: AsyncOpenAI):
    """Build the graph. Nodes use the two MCP sessions and the OpenAI client."""

    async def ask(messages: list[dict], schema):
        """Ask the model for structured output in the shape of `schema`."""
        response = await llm.chat.completions.parse(model=MODEL, messages=messages, response_format=schema)
        return response.choices[0].message.parsed

    # ---- load_profile [model, cached] -------------------------------------
    async def load_profile(state: State) -> dict:
        text = read_resume(RESUME_PATH)
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache = OUTPUTS / ".cache" / f"profile_{key}.json"
        if cache.exists():
            profile = json.loads(cache.read_text())
        else:
            parsed = await ask([{"role": "system", "content": PROFILE_PROMPT},
                                {"role": "user", "content": text}], CandidateProfile)
            profile = parsed.model_dump()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(profile, indent=2))
        print(f"  [load_profile] {profile['name']}: {len(profile['highlights'])} highlights")
        return {"profile": profile}

    # ---- find_domain [MCP research tool + code] ---------------------------
    async def find_domain(state: State) -> dict:
        if state.get("domain"):
            return {"domain": base_domain(state["domain"])}

        # 1. A job link on the company's own site gives the answer directly.
        if state.get("job_url") and is_company_site(base_domain(state["job_url"])):
            domain = base_domain(state["job_url"])
            print(f"  [find_domain] {state['company']} -> {domain} (from the job link)")
            return {"domain": domain}

        # 2. Otherwise search, skip known non-company sites, and prefer a domain
        #    that contains the company's name (datadoghq.com for "Datadog").
        data = tool_json(await research.call_tool(
            "web_search", {"query": f"{state['company']} official website", "max_results": 8}))
        domains = [base_domain(r["url"]) for r in data.get("results", [])]
        domains = [d for d in dict.fromkeys(domains) if is_company_site(d)]
        name = re.sub(r"[^a-z0-9]", "", state["company"].lower())
        matching = [d for d in domains if name and name in d.replace(".", "").replace("-", "")]
        if matching or domains:
            domain = (matching or domains)[0]
            print(f"  [find_domain] {state['company']} -> {domain}  (wrong? re-run with --domain)")
            return {"domain": domain}
        sys.exit(f"Couldn't work out {state['company']}'s website. Re-run with --domain company.com")

    # ---- find_people [MCP people tool] ------------------------------------
    async def find_people(state: State) -> dict:
        # Hunter's free plan returns at most 10 people per search, so search
        # recruiters and engineering leaders separately (1 credit each).
        people_found, cached = {}, True
        for departments in ("hr", "it,management,research"):
            data = tool_json(await people.call_tool(
                "find_people", {"domain": state["domain"], "departments": departments, "limit": 10}))
            if "error" in data:
                sys.exit(f"Hunter: {data['error']}")
            cached = cached and data.get("from_cache", False)
            for p in data["people"]:
                people_found.setdefault(p["email"] or p["name"], p)   # skip duplicates
        candidates = list(people_found.values())
        note = " (from cache, no credits used)" if cached else ""
        print(f"  [find_people] {len(candidates)} people at {state['domain']}{note}")
        return {"candidates": candidates}

    # ---- shortlist [model] ------------------------------------------------
    async def shortlist(state: State) -> dict:
        cands = state["candidates"]
        if not cands:
            return {"shortlist": []}
        listing = "\n".join(f"{i}. {c['name']} | {c['position']} | {c['department']} | {c['seniority']}"
                            for i, c in enumerate(cands))
        result = await ask([
            {"role": "system", "content": SHORTLIST_PROMPT.format(
                company=state["company"], role=state["role"], max_people=MAX_PEOPLE)},
            {"role": "user", "content": listing}], Shortlist)
        picked = []
        for p in result.picks:
            if p.category != "skip" and 0 <= p.index < len(cands) and len(picked) < MAX_PEOPLE:
                picked.append({**cands[p.index], "category": p.category, "why": p.reason})
        print(f"  [shortlist] {len(picked)} suggested")
        return {"shortlist": picked}

    # ---- review [HUMAN: the graph pauses here] ----------------------------
    def review(state: State) -> dict:
        # interrupt() stops the graph and hands this value to main(). When you
        # answer, the graph resumes here and interrupt() returns your answer.
        answer = interrupt({"shortlist": state["shortlist"], "domain": state["domain"]})
        chosen = [state["shortlist"][i] for i in answer if 0 <= i < len(state["shortlist"])]
        return {"selected": chosen}

    def fan_out(state: State):
        """After review: start one research_person per selected person, plus
        research_role, all in parallel (LangGraph's Send API)."""
        if not state.get("selected"):
            return END
        jobs = [Send("research_person", {"person": p, "company": state["company"]}) for p in state["selected"]]
        jobs.append(Send("research_role", state))
        return jobs

    # ---- research_person [MCP research tool + model], one copy per person -
    async def research_person(job: dict) -> dict:
        p, company = job["person"], job["company"]
        queries = [f'"{p["name"]}" {company}',
                   f'"{p["name"]}" {company} talk OR podcast OR blog OR interview OR github']
        results = []
        for q in queries:
            data = tool_json(await research.call_tool("web_search", {"query": q, "max_results": 5}))
            results += data.get("results", [])
        found = await ask([
            {"role": "system", "content": PERSON_PROMPT.format(name=p["name"], position=p["position"], company=company)},
            {"role": "user", "content": json.dumps(results)}], PersonResearch)
        valid_urls = {r["url"] for r in results}
        hooks = [h.model_dump() for h in found.hooks if h.source_url in valid_urls] if found.same_person_confident else []
        print(f"  [research_person] {p['name']}: {len(hooks)} hook(s)")
        return {"research": [{"name": p["name"], "hooks": hooks, "summary": found.summary}]}

    # ---- research_role [MCP research tool + model] ------------------------
    async def research_role(state: State) -> dict:
        material = []
        if state.get("job_url"):
            page = tool_json(await research.call_tool("read_page", {"url": state["job_url"]}))
            if "text" in page:
                material.append({"source": state["job_url"], "text": page["text"]})
        for q in [f'{state["company"]} {state["role"]} job', f'{state["company"]} engineering blog']:
            data = tool_json(await research.call_tool("web_search", {"query": q, "max_results": 5, "time_range": "year"}))
            material += [{"source": r["url"], "text": f'{r["title"]}: {r["snippet"]}'} for r in data.get("results", [])]
        brief = await ask([
            {"role": "system", "content": ROLE_PROMPT.format(role=state["role"], company=state["company"])},
            {"role": "user", "content": json.dumps(material)[:20000]}], RoleBrief)
        print(f"  [research_role] {len(brief.talking_points)} talking point(s)")
        return {"role_brief": brief.model_dump()}

    # ---- draft [model], all people concurrently ---------------------------
    async def draft(state: State) -> dict:
        extra = ""
        if NOTES_PATH.exists():
            extra = f"\nExtra instructions from the candidate (follow them):\n{NOTES_PATH.read_text()}"
        system = DRAFT_PROMPT.format(role=state["role"], note_limit=LINKEDIN_NOTE_LIMIT - 20,
                                     word_limit=EMAIL_WORD_LIMIT, extra=extra)
        research_by_name = {r["name"]: r for r in state.get("research", [])}
        feedback = state.get("feedback", {})
        drafts = dict(state.get("drafts", {}))

        # First pass: draft everyone. Later passes: only people whose draft had problems.
        todo = [p for p in state["selected"] if not drafts or p["name"] in feedback]

        async def one(p: dict):
            context = {
                "recipient": {k: p.get(k) for k in ("name", "position", "category")},
                "company": state["company"], "role": state["role"],
                "person_research": research_by_name.get(p["name"], {}),
                "role_brief": state.get("role_brief", {}),
                "candidate_profile": state["profile"],
            }
            if p["name"] in feedback:
                context["fix_these_problems_from_last_draft"] = feedback[p["name"]]
            d = await ask([{"role": "system", "content": system},
                           {"role": "user", "content": json.dumps(context)}], Draft)
            return p["name"], d.model_dump()

        for name, d in await asyncio.gather(*(one(p) for p in todo)):
            drafts[name] = d
        print(f"  [draft] {len(todo)} draft(s) written")
        return {"drafts": drafts}

    # ---- check [code + model] ---------------------------------------------
    async def check(state: State) -> dict:
        research_by_name = {r["name"]: r for r in state.get("research", [])}
        feedback = {}

        async def one(name: str, d: dict):
            problems = []
            if len(d["linkedin_note"]) > LINKEDIN_NOTE_LIMIT:                      # code checks
                problems.append(f"LinkedIn note is {len(d['linkedin_note'])} characters; max {LINKEDIN_NOTE_LIMIT}.")
            if len(d["email_body"].split()) > EMAIL_WORD_LIMIT + 20:
                problems.append(f"Email is too long ({len(d['email_body'].split())} words).")
            if re.search(r"\[[^\]]*\]|\{[^}]*\}", d["email_body"] + d["linkedin_note"]):
                problems.append("Contains a placeholder like [Company] or {name}.")
            verdict = await ask([{"role": "system", "content": CHECK_PROMPT},                # model check
                                 {"role": "user", "content": json.dumps({
                                     "draft": d, "candidate_profile": state["profile"],
                                     "person_research": research_by_name.get(name, {}),
                                     "role_brief": state.get("role_brief", {})})}], FactCheck)
            return name, problems + verdict.problems

        for name, problems in await asyncio.gather(*(one(n, d) for n, d in state["drafts"].items()
                                                     if not state.get("feedback") or n in state["feedback"])):
            if problems:
                feedback[name] = problems
        print(f"  [check] {len(feedback)} draft(s) need fixes")
        return {"feedback": feedback, "revisions": state.get("revisions", 0) + 1}

    def after_check(state: State) -> str:
        if state["feedback"] and state["revisions"] < 2:
            return "draft"                  # redraft only the ones with problems (once)
        return "verify_emails"

    # ---- verify_emails [MCP people tool] ----------------------------------
    async def verify_emails(state: State) -> dict:
        updated = []
        for p in state["selected"]:
            p = dict(p)
            if p.get("email") and p.get("email_status") != "valid":
                p["email_status"] = tool_json(await people.call_tool("verify_email", {"email": p["email"]})).get("status")
            updated.append(p)
        print(f"  [verify_emails] {sum(p.get('email_status') == 'valid' for p in updated)}/{len(updated)} valid")
        return {"selected": updated}

    # ---- save [code] ------------------------------------------------------
    def save(state: State) -> dict:
        OUTPUTS.mkdir(exist_ok=True)
        path = OUTPUTS / f"{date.today()}_{slug(state['company'])}_{slug(state['role'])}.md"
        research_by_name = {r["name"]: r for r in state.get("research", [])}
        lines = [f"# Outreach: {state['role']} at {state['company']}", f"_Generated {date.today()}. Review and edit before sending._", ""]
        brief = state.get("role_brief", {})
        if brief:
            lines += ["## About the role", brief.get("summary", ""), ""]
        for p in state["selected"]:
            d = state["drafts"].get(p["name"], {})
            r = research_by_name.get(p["name"], {})
            lines += [f"## {p['name']}, {p.get('position')} ({p['category'].replace('_', ' ')})",
                      f"- LinkedIn: {p.get('linkedin') or 'not found'}",
                      f"- Email: {p.get('email') or 'not found'} ({p.get('email_status') or 'unverified'})", ""]
            lines += ["**Research**"] + ([f"- {h['fact']} ([source]({h['source_url']}))" for h in r.get("hooks", [])]
                                         or ["- Nothing specific found; message is based on the role and team."]) + [""]
            lines += ["**LinkedIn note**", "```", d.get("linkedin_note", ""), "```",
                      f"**Email**: {d.get('email_subject', '')}", "```", d.get("email_body", ""), "```"]
            if state.get("feedback", {}).get(p["name"]):
                lines += ["> ⚠️ The checker still flagged: " + "; ".join(state["feedback"][p["name"]])]
            lines.append("")
        path.write_text("\n".join(lines))

        tracker = OUTPUTS / "tracker.csv"
        new = not tracker.exists()
        with tracker.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "company", "role", "name", "title", "type", "email", "email_status", "linkedin", "status"])
            for p in state["selected"]:
                w.writerow([date.today(), state["company"], state["role"], p["name"], p.get("position"),
                            p["category"], p.get("email"), p.get("email_status"), p.get("linkedin"), "drafted"])
        print(f"  [save] outputs/{path.name} and outputs/tracker.csv")
        return {"log": [f"outputs/{path.name}"]}

    # =======================================================================
    # 5. EDGES
    # =======================================================================
    g = StateGraph(State)
    for name, fn in [("load_profile", load_profile), ("find_domain", find_domain), ("find_people", find_people),
                     ("shortlist", shortlist), ("review", review), ("research_person", research_person),
                     ("research_role", research_role), ("draft", draft), ("check", check),
                     ("verify_emails", verify_emails), ("save", save)]:
        g.add_node(name, fn)

    g.add_edge(START, "load_profile")
    g.add_edge("load_profile", "find_domain")
    g.add_edge("find_domain", "find_people")
    g.add_edge("find_people", "shortlist")
    g.add_edge("shortlist", "review")
    g.add_conditional_edges("review", fan_out, ["research_person", "research_role", END])
    g.add_edge("research_person", "draft")       # draft waits for ALL parallel research to finish
    g.add_edge("research_role", "draft")
    g.add_edge("draft", "check")
    g.add_conditional_edges("check", after_check, ["draft", "verify_emails"])
    g.add_edge("verify_emails", "save")
    g.add_edge("save", END)

    # A checkpointer is required for interrupt(): it saves the state while the graph is paused.
    return g.compile(checkpointer=InMemorySaver())


# ===========================================================================
# 6. MAIN: start both MCP servers, run the graph, handle your review
# ===========================================================================
def show_shortlist(info: dict) -> None:
    print(f"\nSuggested people at {info['domain']}:")
    for i, p in enumerate(info["shortlist"], 1):
        email = f"{p.get('email')} ({p.get('email_confidence')}%)" if p.get("email") else "no email"
        print(f"  {i}. {p['name']}, {p['position']} [{p['category'].replace('_', ' ')}]")
        print(f"     {email} | {p.get('linkedin') or 'no LinkedIn'} | why: {p['why']}")


def ask_selection(count: int) -> list[int]:
    raw = input("\nWho should I research and draft for? (e.g. 1,3,4 / all / none): ").strip().lower()
    if raw in {"all", "a", ""}:
        return list(range(count))
    if raw in {"none", "n", "q"}:
        return []
    return [int(x) - 1 for x in re.findall(r"\d+", raw)]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare personalized cold outreach for a job.")
    parser.add_argument("--company")
    parser.add_argument("--role")
    parser.add_argument("--domain", help="company website, e.g. stripe.com (skips the lookup)")
    parser.add_argument("--job-url", help="link to the job posting (optional, improves the messages)")
    parser.add_argument("--graph", action="store_true", help="print the graph as a Mermaid diagram")
    args = parser.parse_args()

    if args.graph:
        print(build_graph(None, None, None).get_graph().draw_mermaid())
        return
    company = args.company or input("Company: ").strip()
    role = args.role or input("Role: ").strip()

    servers = {name: StdioServerParameters(command=sys.executable, args=[str(ROOT / "servers" / f"{name}_server.py")])
               for name in ("people", "research")}

    async with AsyncExitStack() as stack:
        sessions = {}
        for name, params in servers.items():          # start and connect both MCP servers
            read, write = await stack.enter_async_context(stdio_client(params))
            sessions[name] = await stack.enter_async_context(ClientSession(read, write))
            await sessions[name].initialize()
        print("Connected to MCP servers: people (Hunter), research (Tavily)\n")

        app = build_graph(sessions["people"], sessions["research"], AsyncOpenAI())
        config = {"configurable": {"thread_id": f"{slug(company)}-{slug(role)}"}}
        inputs = {"company": company, "role": role, "domain": args.domain or "", "job_url": args.job_url or ""}

        result = await app.ainvoke(inputs, config)
        while "__interrupt__" in result:              # the graph paused at review
            info = result["__interrupt__"][0].value
            if not info["shortlist"]:
                print("\nNo suitable recruiters or engineering managers found. Try --domain, or a bigger company.")
                return
            show_shortlist(info)
            result = await app.ainvoke(Command(resume=ask_selection(len(info["shortlist"]))), config)

        if result.get("log"):
            print(f"\nDone. Open {result['log'][-1]} to review, edit and send the messages yourself.")
        else:
            print("\nNothing selected, nothing drafted.")


if __name__ == "__main__":
    asyncio.run(main())
