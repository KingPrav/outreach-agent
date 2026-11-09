"""Graph state and the structured-output schemas the model must return."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from .config import EMAIL_WORD_LIMIT, LINKEDIN_NOTE_LIMIT


class State(TypedDict, total=False):
    company: str
    role: str
    domain: str                                     # email domain people were found at
    domain_candidates: list[str]
    job_url: str
    profile: dict                                   # the candidate's resume, structured
    candidates: list[dict]                          # everyone the people source returned
    shortlist: list[dict]                           # model-suggested contacts
    selected: list[dict]                            # contacts the user approved
    role_brief: dict
    research: Annotated[list[dict], operator.add]   # parallel research_person branches append
    drafts: dict                                    # name -> draft
    feedback: dict                                  # name -> problems from the last check
    revisions: int
    report_path: str
    metrics: Annotated[list[dict], operator.add]    # one row per node run: time, model calls, tokens


class Highlight(BaseModel):
    title: str = Field(description="Project or job, e.g. 'HireRank: Resume Matching Service'")
    detail: str = Field(description="One sentence on what was built or done, using only facts from the resume")
    metrics: list[str] = Field(description="Exact numbers from the resume for this item, if any")


class CandidateProfile(BaseModel):
    name: str
    headline: str
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
    artifact: str = Field(description="The exact title or name of the talk, post, podcast episode, repository or "
                                      "launch, copied word for word from the search result's title or snippet")
    fact: str = Field(description="One sentence: what the person did with that artifact, e.g. 'Wrote the post \"X\"'")
    source_url: str = Field(description="The URL of the search result the artifact appears in")


class PersonResearch(BaseModel):
    same_person_confident: bool = Field(description="True only if the results clearly refer to this person at this company")
    hooks: list[Hook] = Field(description="Up to 3 hooks. Empty if nothing specific and reliable was found.")
    summary: str = Field(description="One sentence on what this person works on, or '' if unknown")


class RoleBrief(BaseModel):
    summary: str = Field(description="2-3 sentences on the role and team")
    key_requirements: list[str]
    talking_points: list[str] = Field(description="Specific facts worth mentioning, each with its source URL in brackets")


class Draft(BaseModel):
    linkedin_note: str = Field(description=f"Connection note, at most {LINKEDIN_NOTE_LIMIT - 20} characters")
    email_subject: str
    email_body: str = Field(description=f"Plain-text email, at most {EMAIL_WORD_LIMIT} words, signed with the candidate's name")
    hook_used: str = Field(description="Which fact about the person or company the message is built on, or 'none'")
    resume_points_used: list[str] = Field(description="Which resume highlights the message mentions")


class FactCheck(BaseModel):
    problems: list[str] = Field(description="Claims not supported by the profile, research or role brief; generic filler; wrong names. Empty if fine.")
