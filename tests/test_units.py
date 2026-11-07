from datetime import date

import pytest

from outreach_agent.checks import rule_problems
from outreach_agent.cli import parse_selection
from outreach_agent.domains import base_domain, is_company_site, rank_domains, slug
from outreach_agent.report import write_report

from .fakes import clean_draft


@pytest.mark.parametrize("given, expected", [
    ("https://careers.datadoghq.com/jobs/1", "datadoghq.com"),
    ("www.Stripe.com", "stripe.com"),
    ("https://shop.company.co.uk/x", "company.co.uk"),
    ("http://localhost:8000", "localhost"),
    ("", ""),
])
def test_base_domain(given, expected):
    assert base_domain(given) == expected


def test_job_boards_are_not_company_sites():
    assert not is_company_site("boards.greenhouse.io")
    assert not is_company_site("linkedin.com")
    assert is_company_site("northwind.dev")


def test_rank_domains_prefers_exact_name_and_drops_noise():
    got = rank_domains("ByteDance", ["other.com", "joinbytedance.com", "bytedance.com", "linkedin.com", "bytedance.com"])
    assert got == ["bytedance.com", "joinbytedance.com", "other.com"]


def test_slug():
    assert slug("Machine Learning Engineer, II") == "machine-learning-engineer-ii"


def test_clean_draft_has_no_rule_problems():
    assert rule_problems(clean_draft("Jordan Lee").model_dump(), "Jordan Lee", "Northwind") == []


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.update(linkedin_note="x" * 301), "300"),
    (lambda d: d.update(email_body=d["email_body"] + " word" * 200), "too long"),
    (lambda d: d.update(email_body=d["email_body"] + " {name}"), "placeholder"),
    (lambda d: d.update(email_body="I hope this email finds you well. " + d["email_body"]), "Stock phrase"),
])
def test_rule_problems_catch_each_mistake(mutate, expected):
    d = clean_draft("Jordan Lee").model_dump()
    mutate(d)
    assert any(expected in p for p in rule_problems(d, "Jordan Lee", "Northwind"))


def test_rule_problems_flag_wrong_recipient_and_company():
    d = clean_draft("Jordan Lee").model_dump()
    problems = rule_problems(d, "Casey Morgan", "Acme")
    assert any("Casey Morgan" in p for p in problems) and any("Acme" in p for p in problems)


@pytest.mark.parametrize("raw, expected", [
    ("1,3", [0, 2]), ("all", [0, 1, 2]), ("", [0, 1, 2]), ("none", []), ("2 and 9", [1]),
])
def test_parse_selection(raw, expected):
    assert parse_selection(raw, 3) == expected


def test_tracker_header_written_once(tmp_path):
    state = {"company": "Northwind", "role": "AI Engineer", "research": [], "drafts": {},
             "selected": [{"name": "Jordan Lee", "position": "EM", "category": "engineering_manager"}]}
    write_report(state, tmp_path, date(2026, 11, 7))
    path = write_report(state, tmp_path, date(2026, 11, 7))
    lines = (tmp_path / "tracker.csv").read_text().splitlines()
    assert lines[0].startswith("date,") and len(lines) == 3
    assert path.name == "2026-11-07_northwind_ai-engineer.md"
