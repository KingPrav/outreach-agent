"""The outreach graph.

    load_profile -> find_domain -> find_people -> shortlist -> review (human)
        -> research_person (one per contact, parallel) + research_role
        -> draft -> check --(problems, once)--> draft
        -> verify_emails -> save
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from . import prompts
from .checks import rule_problems
from .clients import AgentError, StructuredLLM, Tools
from .config import EMAIL_WORD_LIMIT, LINKEDIN_NOTE_LIMIT, MAX_REVISIONS, Settings
from .domains import base_domain, is_company_site, rank_domains
from .report import write_report
from .schemas import (CandidateProfile, Draft, FactCheck, PersonResearch, RoleBrief, Shortlist,
                      State)

# Hunter errors that no other domain attempt can fix.
FATAL_PEOPLE_ERRORS = ("401", "403", "429", "not set", "usage limit")


def read_resume(path: Path) -> str:
    if not path.exists():
        raise AgentError(f"Resume not found at {path}. Put it in profile/ and set RESUME_PATH in .env.")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text()


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def grounded_hooks(found: PersonResearch, results: list[dict]) -> list[dict]:
    """Keep a hook only if the model is sure it found the right person, the cited URL was actually
    retrieved, and the artifact's title appears in that result's title or snippet."""
    if not found.same_person_confident:
        return []
    text_by_url = {r["url"]: _norm(f'{r.get("title", "")} {r.get("snippet", "")}') for r in results}
    kept = []
    for h in found.hooks:
        artifact = _norm(h.artifact)
        if h.source_url in text_by_url and len(artifact) >= 3 and artifact in text_by_url[h.source_url]:
            kept.append(h.model_dump())
    return kept


def build_graph(people: Tools, research: Tools, llm: StructuredLLM, settings: Settings,
                checkpointer=None, log=print):

    async def load_profile(state: State) -> dict:
        text = read_resume(settings.resume_path)
        cache = settings.cache_dir / f"profile_{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"
        if cache.exists():
            profile = json.loads(cache.read_text())
        else:
            parsed = await llm.parse([{"role": "system", "content": prompts.PROFILE_PROMPT},
                                      {"role": "user", "content": text}], CandidateProfile)
            profile = parsed.model_dump()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(profile, indent=2))
        log(f"  [load_profile] {profile['name']}: {len(profile['highlights'])} highlights")
        return {"profile": profile}

    async def find_domain(state: State) -> dict:
        # Careers sites (joinbytedance.com) often differ from the email domain, so we keep
        # a few ranked fallbacks rather than trusting one guess.
        if state.get("domain"):
            return {"domain_candidates": [base_domain(state["domain"])]}
        found = []
        if state.get("job_url") and is_company_site(base_domain(state["job_url"])):
            found.append(base_domain(state["job_url"]))
        data = await research.call("web_search", {"query": f"{state['company']} official website", "max_results": 8})
        found += [base_domain(r["url"]) for r in data.get("results", [])]
        candidates = rank_domains(state["company"], found)
        log(f"  [find_domain] fallback domains: {candidates or 'none'}")
        return {"domain_candidates": candidates}

    async def find_people(state: State) -> dict:
        # Hunter resolves a company name to its email domain better than we can, so try the
        # name first, then our fallbacks. --domain means: use exactly that.
        if state.get("domain"):
            attempts = [{"domain": d} for d in state["domain_candidates"]]
        else:
            attempts = [{"company": state["company"]}] + [{"domain": d} for d in state["domain_candidates"]]

        last_error = ""
        for target in attempts[:3]:
            found, domain, failed = {}, target.get("domain", ""), False
            # The free plan caps results at 10 per search, so recruiters and engineering
            # leaders are searched separately.
            for departments in ("hr", "it,management,research"):
                data = await people.call("find_people", {**target, "departments": departments, "limit": 10})
                if "error" in data:
                    if any(code in data["error"] for code in FATAL_PEOPLE_ERRORS):
                        raise AgentError(f"Hunter: {data['error']}")
                    last_error, failed = data["error"], True
                    break
                domain = data.get("domain") or domain
                for p in data.get("people", []):
                    found.setdefault(p.get("email") or p["name"], p)
            label = target.get("domain") or f"company name '{target['company']}'"
            log(f"  [find_people] {len(found)} people via {label}" + (f" ({last_error})" if failed else ""))
            if found:
                return {"candidates": list(found.values()), "domain": domain}
        return {"candidates": [], "domain": state.get("domain", "")}

    async def shortlist(state: State) -> dict:
        cands = state.get("candidates", [])
        if not cands:
            return {"shortlist": []}
        listing = "\n".join(f"{i}. {c['name']} | {c.get('position')} | {c.get('department')} | {c.get('seniority')}"
                            for i, c in enumerate(cands))
        result = await llm.parse([
            {"role": "system", "content": prompts.SHORTLIST_PROMPT.format(
                company=state["company"], role=state["role"], max_people=settings.max_people)},
            {"role": "user", "content": listing}], Shortlist)
        picked, seen = [], set()
        for p in result.picks:
            if p.category == "skip" or not 0 <= p.index < len(cands) or p.index in seen:
                continue
            seen.add(p.index)
            picked.append({**cands[p.index], "category": p.category, "why": p.reason})
            if len(picked) == settings.max_people:
                break
        log(f"  [shortlist] {len(picked)} suggested")
        return {"shortlist": picked}

    def review(state: State) -> dict:
        if not state.get("shortlist"):
            return {"selected": []}
        answer = interrupt({"shortlist": state["shortlist"], "domain": state.get("domain", "")})
        chosen = [state["shortlist"][i] for i in dict.fromkeys(answer) if 0 <= i < len(state["shortlist"])]
        return {"selected": chosen}

    def fan_out(state: State):
        if not state.get("selected"):
            return END
        jobs = [Send("research_person", {"person": p, "company": state["company"]}) for p in state["selected"]]
        jobs.append(Send("research_role", state))
        return jobs

    async def research_person(job: dict) -> dict:
        p, company = job["person"], job["company"]
        queries = [f'"{p["name"]}" {company}',
                   f'"{p["name"]}" {company} talk OR podcast OR blog OR interview OR github']
        results = []
        for data in await asyncio.gather(*(research.call("web_search", {"query": q, "max_results": 5})
                                           for q in queries)):
            results += data.get("results", [])
        found = await llm.parse([
            {"role": "system", "content": prompts.PERSON_PROMPT.format(
                name=p["name"], position=p.get("position"), company=company)},
            {"role": "user", "content": json.dumps(results)}], PersonResearch)
        hooks = grounded_hooks(found, results)
        log(f"  [research_person] {p['name']}: {len(hooks)} hook(s)")
        return {"research": [{"name": p["name"], "hooks": hooks, "summary": found.summary}]}

    async def research_role(state: State) -> dict:
        material = []
        if state.get("job_url"):
            page = await research.call("read_page", {"url": state["job_url"]})
            if "text" in page:
                material.append({"source": state["job_url"], "text": page["text"]})
        for q in (f'{state["company"]} {state["role"]} job', f'{state["company"]} engineering blog'):
            data = await research.call("web_search", {"query": q, "max_results": 5, "time_range": "year"})
            material += [{"source": r["url"], "text": f'{r["title"]}: {r["snippet"]}'} for r in data.get("results", [])]
        brief = await llm.parse([
            {"role": "system", "content": prompts.ROLE_PROMPT.format(role=state["role"], company=state["company"])},
            {"role": "user", "content": json.dumps(material)[:20000]}], RoleBrief)
        log(f"  [research_role] {len(brief.talking_points)} talking point(s)")
        return {"role_brief": brief.model_dump()}

    async def draft(state: State) -> dict:
        extra = ""
        if settings.notes_path.exists():
            extra = f"\nExtra instructions from the candidate (follow them):\n{settings.notes_path.read_text()}"
        system = prompts.DRAFT_PROMPT.format(role=state["role"], note_limit=LINKEDIN_NOTE_LIMIT - 20,
                                             word_limit=EMAIL_WORD_LIMIT, extra=extra)
        research_by_name = {r["name"]: r for r in state.get("research", [])}
        feedback = state.get("feedback", {})
        drafts = dict(state.get("drafts", {}))
        # First pass drafts everyone; a redraft only touches drafts that failed the check.
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
            d = await llm.parse([{"role": "system", "content": system},
                                 {"role": "user", "content": json.dumps(context)}], Draft)
            return p["name"], d.model_dump()

        for name, d in await asyncio.gather(*(one(p) for p in todo)):
            drafts[name] = d
        log(f"  [draft] {len(todo)} draft(s) written")
        return {"drafts": drafts}

    async def check(state: State) -> dict:
        research_by_name = {r["name"]: r for r in state.get("research", [])}
        previous = state.get("feedback")
        to_check = {n: d for n, d in state["drafts"].items() if not previous or n in previous}

        async def one(name: str, d: dict):
            problems = rule_problems(d, recipient_name=name, company=state["company"])
            verdict = await llm.parse([
                {"role": "system", "content": prompts.CHECK_PROMPT},
                {"role": "user", "content": json.dumps({
                    "draft": d, "candidate_profile": state["profile"],
                    "person_research": research_by_name.get(name, {}),
                    "role_brief": state.get("role_brief", {})})}], FactCheck)
            return name, problems + verdict.problems

        results = await asyncio.gather(*(one(n, d) for n, d in to_check.items()))
        feedback = {name: problems for name, problems in results if problems}
        log(f"  [check] {len(feedback)} draft(s) need fixes")
        return {"feedback": feedback, "revisions": state.get("revisions", 0) + 1}

    def after_check(state: State) -> str:
        return "draft" if state["feedback"] and state["revisions"] < MAX_REVISIONS else "verify_emails"

    async def verify_emails(state: State) -> dict:
        async def one(p: dict) -> dict:
            p = dict(p)
            if p.get("email") and p.get("email_status") != "valid":
                p["email_status"] = (await people.call("verify_email", {"email": p["email"]})).get("status")
            return p

        updated = list(await asyncio.gather(*(one(p) for p in state["selected"])))
        log(f"  [verify_emails] {sum(p.get('email_status') == 'valid' for p in updated)}/{len(updated)} valid")
        return {"selected": updated}

    def save(state: State) -> dict:
        path = write_report(state, settings.outputs)
        log(f"  [save] {path.relative_to(settings.outputs.parent)} and outputs/tracker.csv")
        return {"report_path": str(path)}

    g = StateGraph(State)
    for fn in (load_profile, find_domain, find_people, shortlist, review, research_person,
               research_role, draft, check, verify_emails, save):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "load_profile")
    g.add_edge("load_profile", "find_domain")
    g.add_edge("find_domain", "find_people")
    g.add_edge("find_people", "shortlist")
    g.add_edge("shortlist", "review")
    g.add_conditional_edges("review", fan_out, ["research_person", "research_role", END])
    g.add_edge(["research_person", "research_role"], "draft")   # waits for every branch
    g.add_edge("draft", "check")
    g.add_conditional_edges("check", after_check, ["draft", "verify_emails"])
    g.add_edge("verify_emails", "save")
    g.add_edge("save", END)
    return g.compile(checkpointer=checkpointer)
