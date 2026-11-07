"""Prompts. Kept separate so they can be versioned and evaluated on their own."""

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
