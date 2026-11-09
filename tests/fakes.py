"""In-memory stand-ins for the model and the MCP servers. No network, no keys."""

import json
from collections import Counter

from outreach_agent.clients import record_usage

from outreach_agent.schemas import (CandidateProfile, Draft, FactCheck, Hook, PersonResearch, Pick, RoleBrief,
                                    Shortlist)

PEOPLE = [
    {"name": "Jordan Lee", "position": "Engineering Manager, ML Platform", "department": "it", "seniority": "senior",
     "email": "jordan@northwind.dev", "email_confidence": 95, "email_status": None, "linkedin": None},
    {"name": "Sam Patel", "position": "Technical Recruiter", "department": "hr", "seniority": "junior",
     "email": "sam@northwind.dev", "email_confidence": 90, "email_status": "valid", "linkedin": None},
    {"name": "Riley Chen", "position": "Account Executive", "department": "sales", "seniority": "junior",
     "email": "riley@northwind.dev", "email_confidence": 80, "email_status": None, "linkedin": None},
]


def clean_draft(name: str, company: str = "Northwind") -> Draft:
    first = name.split()[0]
    return Draft(
        linkedin_note=f"Hi {first}, I saw {company}'s post on eval tooling and built a RAG eval harness myself. Would love to connect.",
        email_subject="New grad AI engineer, eval harness experience",
        email_body=f"Hi {first},\n\nYour team's post on eval tooling at {company} caught my eye. I built a RAG "
                   "eval harness with a CI gate that blocks regressions. Would you be open to a 15 minute chat?\n\nAlex",
        hook_used="team post on eval tooling",
        resume_points_used=["RAG eval harness"],
    )


class FakeLLM:
    """Answers each schema with canned output. `bad_first_draft` names get a placeholder
    in their first draft, so the rule check fails once and forces a redraft."""

    def __init__(self, bad_first_draft=(), unconfident=(), picks=None):
        self.calls = Counter()
        self.bad_first_draft = set(bad_first_draft)
        self.unconfident = set(unconfident)
        self.picks = picks
        self.drafted = Counter()
        self.messages = {}

    async def parse(self, messages, schema):
        self.calls[schema.__name__] += 1
        self.messages.setdefault(schema.__name__, []).append(messages)
        record_usage(prompt=100, completion=20, cached=40)
        system, user = messages[0]["content"], messages[-1]["content"]
        if schema is CandidateProfile:
            return CandidateProfile(name="Alex Rivera", headline="MS CS student", education=["MS CS"],
                                    skills=["Python"], highlights=[])
        if schema is Shortlist:
            picks = self.picks if self.picks is not None else [
                Pick(index=0, category="recruiter", reason="recruiter"),
                Pick(index=1, category="engineering_manager", reason="ML platform"),
                Pick(index=2, category="skip", reason="sales")]
            return Shortlist(picks=picks)
        if schema is PersonResearch:
            name = next(p["name"] for p in PEOPLE if p["name"] in system)
            results = json.loads(user)
            return PersonResearch(
                same_person_confident=name not in self.unconfident,
                hooks=[Hook(artifact="eval tooling", fact="Spoke about eval tooling", source_url=results[0]["url"]),
                       Hook(artifact="eval tooling", fact="Invented fact", source_url="https://not-in-results.example"),
                       Hook(artifact="Keynote at NeurIPS", fact="Unsupported title", source_url=results[0]["url"])],
                summary="Leads the ML platform team")
        if schema is RoleBrief:
            return RoleBrief(summary="Builds agents.", key_requirements=["Python"],
                             talking_points=["Eval tooling [https://northwind.dev/blog]"])
        if schema is Draft:
            name = json.loads(user)["recipient"]["name"]
            self.drafted[name] += 1
            d = clean_draft(name)
            if name in self.bad_first_draft and self.drafted[name] == 1:
                d.email_body += " [Company]"
            return d
        if schema is FactCheck:
            return FactCheck(problems=[])
        raise AssertionError(f"unexpected schema {schema}")


class FakeTools:
    def __init__(self, handlers):
        self.handlers, self.calls = handlers, []

    async def call(self, name, args):
        self.calls.append((name, args))
        return self.handlers[name](args)


def people_tools(error_for_company=None):
    def find_people(args):
        if "company" in args and error_for_company:
            return {"error": error_for_company}
        depts = args["departments"]
        found = [p for p in PEOPLE if (p["department"] == "hr") == (depts == "hr")]
        return {"domain": "northwind.dev", "people": found}

    return FakeTools({
        "find_people": find_people,
        "verify_email": lambda a: {"email": a["email"], "status": "valid"},
    })


def research_tools():
    def web_search(args):
        q = args["query"]
        return {"results": [{"title": "Result", "url": f"https://example.com/{abs(hash(q)) % 10_000}",
                             "snippet": "about eval tooling"},
                            {"title": "Northwind", "url": "https://www.northwind.dev/about", "snippet": "company"}]}

    return FakeTools({"web_search": web_search,
                      "read_page": lambda a: {"url": a["url"], "text": "Job posting text"}})
