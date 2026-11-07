"""Seeded-error dataset for the draft checker.

Six accurate drafts (fictional candidate, people and company) plus one copy of each with a
single known error inserted. A good checker flags every seeded copy and leaves the
accurate originals alone. Every fact in the clean drafts is backed by PROFILE, RESEARCH or
ROLE_BRIEF, so any flag on a clean draft is a false positive.
"""

import copy

COMPANY = "Northwind Labs"
COMPETITOR = "Contoso AI"

PROFILE = {
    "name": "Alex Rivera",
    "headline": "MS CS student at State University (May 2027), AI engineer",
    "education": ["MS Computer Science, State University, May 2027", "BS Computer Science, 2024"],
    "skills": ["Python", "LangGraph", "FastAPI", "PostgreSQL", "Docker", "PyTorch"],
    "highlights": [
        {"title": "Docs Assistant: evaluated RAG system",
         "detail": "Built a RAG assistant with an eval harness and a CI gate that blocks quality regressions.",
         "metrics": ["held out pass rate 87% to 97%", "hit@1 43% to 80%"]},
        {"title": "Software Engineer Intern, Acme Logistics",
         "detail": "Added Prometheus metrics and Grafana dashboards to production services.",
         "metrics": ["incident response time cut by 25%"]},
    ],
}

RESEARCH = {
    "Jordan Lee": {"summary": "Leads the agent evaluation team.",
                   "hooks": [{"fact": "Wrote a blog post on regression testing for LLM agents",
                              "source_url": "https://northwind.example/blog/agent-regressions"}]},
    "Sam Patel": {"summary": "Recruits for early career engineering roles.", "hooks": []},
    "Morgan Diaz": {"summary": "Engineering manager for the retrieval platform.",
                    "hooks": [{"fact": "Gave a talk on hybrid search at a 2026 search meetup",
                               "source_url": "https://meetup.example/talks/hybrid-search"}]},
}

ROLE_BRIEF = {
    "summary": "New grad AI engineer on the agent platform team, building evals and retrieval for customer agents.",
    "key_requirements": ["Python", "LLM evaluation", "retrieval"],
    "talking_points": ["Northwind Labs open sourced its agent eval runner in 2026 [https://northwind.example/oss]"],
}

CLEAN = [
    ("Jordan Lee", "Hi Jordan,\n\nYour post on regression testing for LLM agents matched a problem I worked on: my RAG "
     "docs assistant has a CI gate that blocks quality regressions, and it raised held out pass rate from 87% to 97%. "
     "I'm applying for the new grad AI engineer role at Northwind Labs. Would you be open to a 15 minute chat about "
     "the team's work?\n\nAlex Rivera"),
    ("Jordan Lee", "Hi Jordan,\n\nI read your blog post on regression testing for agents. I'm finishing an MS in CS "
     "and built an eval harness for a RAG system, with a CI gate that blocks regressions. I'd value your advice on "
     "applying to the AI engineer role on the Northwind Labs agent platform team.\n\nAlex Rivera"),
    ("Sam Patel", "Hi Sam,\n\nI'm applying for the new grad AI engineer role at Northwind Labs. I built an evaluated "
     "RAG system (hit@1 from 43% to 80%) and, as an intern at Acme Logistics, added Prometheus and Grafana monitoring "
     "that cut incident response time by 25%. Are you the right person to talk to about this role?\n\nAlex Rivera"),
    ("Sam Patel", "Hi Sam,\n\nI saw Northwind Labs open sourced its agent eval runner this year. Evals are the core "
     "of my own work: my RAG assistant ships behind a CI quality gate. Could we have a short chat about fit for the "
     "new grad AI engineer role?\n\nAlex Rivera"),
    ("Morgan Diaz", "Hi Morgan,\n\nYour talk on hybrid search caught my attention because I tested hybrid search on "
     "my own RAG system. I'm applying for the AI engineer role at Northwind Labs, which covers retrieval for customer "
     "agents. Would you have 15 minutes to talk about the retrieval platform?\n\nAlex Rivera"),
    ("Morgan Diaz", "Hi Morgan,\n\nI'm a CS master's student applying to the new grad AI engineer role at Northwind "
     "Labs. My RAG project improved hit@1 from 43% to 80%, and I'd love to learn how your team approaches retrieval. "
     "Would you be open to a short chat?\n\nAlex Rivera"),
]

# Each mutation inserts exactly one unsupported or wrong statement.
MUTATIONS = {
    "fabricated_metric": lambda body: _insert(body, "I also cut its p95 latency by 60% with a Redis cache."),
    "fabricated_skill": lambda body: _insert(body, "I've also shipped production services in Rust and Go."),
    "fabricated_experience": lambda body: _insert(body, "Last summer I interned on the ML team at Google."),
    "fake_familiarity": lambda body: _insert(body, "I've followed your work closely for years."),
    "invented_recipient_fact": lambda body: _insert(body, "Congratulations on your recent promotion to director."),
    "wrong_company": lambda body: body.replace(COMPANY, COMPETITOR),
}


def _insert(body: str, sentence: str) -> str:
    head, sep, sign_off = body.rpartition("\n\n")
    return f"{head} {sentence}{sep}{sign_off}"


def _draft(name: str, body: str) -> dict:
    return {"linkedin_note": f"Hi {name.split()[0]}, I'm applying to the AI engineer role at {COMPANY}. Would love to connect.",
            "email_subject": "New grad AI engineer role", "email_body": body,
            "hook_used": "see email", "resume_points_used": []}


def build_cases() -> list[dict]:
    cases = []
    for i, (name, body) in enumerate(CLEAN):
        context = {"candidate_profile": PROFILE, "person_research": RESEARCH[name], "role_brief": ROLE_BRIEF}
        cases.append({"id": f"clean-{i}", "recipient": name, "error": None, "draft": _draft(name, body), **context})
        for kind, mutate in MUTATIONS.items():
            mutated = mutate(body)
            assert mutated != body, f"{kind} did not change clean-{i}"
            draft = _draft(name, mutated)
            if kind == "wrong_company":
                draft["linkedin_note"] = draft["linkedin_note"].replace(COMPANY, COMPETITOR)
            cases.append({"id": f"{kind}-{i}", "recipient": name, "error": kind, "draft": draft,
                          **copy.deepcopy(context)})
    return cases
