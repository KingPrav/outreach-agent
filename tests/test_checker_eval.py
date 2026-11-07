from evals.checker_dataset import CLEAN, MUTATIONS, build_cases
from evals.run_checker_eval import check_case, stability, summarize
from outreach_agent.checks import rule_problems
from outreach_agent.schemas import FactCheck


def test_dataset_shape():
    cases = build_cases()
    assert len(cases) == len(CLEAN) * (1 + len(MUTATIONS))
    assert len({c["id"] for c in cases}) == len(cases)


def test_clean_drafts_pass_the_rule_checks():
    # Otherwise the false positive rate would measure the dataset, not the checker.
    for c in build_cases():
        if c["error"] is None:
            assert rule_problems(c["draft"], c["recipient"], "Northwind Labs") == [], c["id"]


class FlagsSeededText:
    """Stand-in model that flags any draft containing one of the seeded sentences."""
    markers = ("60%", "Rust", "Google", "for years", "promotion")

    async def parse(self, messages, schema):
        body = messages[-1]["content"]
        return FactCheck(problems=["unsupported"] if any(m in body for m in self.markers) else [])


async def test_scoring_with_a_perfect_checker():
    rows = [await check_case(c, FlagsSeededText(), rules_only=False) for c in build_cases()]
    s = summarize(rows)
    assert s["recall"] == 1.0 and s["false_positive_rate"] == 0.0


async def test_rules_alone_only_catch_the_wrong_company():
    rows = [await check_case(c, None, rules_only=True) for c in build_cases()]
    s = summarize(rows)["by_kind"]
    assert s["wrong_company"]["rate"] == 1.0
    assert s["fabricated_metric"]["rate"] == 0.0 and s["clean"]["rate"] == 0.0


def test_stability():
    a = [{"id": "x", "flagged": True}, {"id": "y", "flagged": False}]
    b = [{"id": "x", "flagged": True}, {"id": "y", "flagged": True}]
    assert stability([a, b]) == 0.5
