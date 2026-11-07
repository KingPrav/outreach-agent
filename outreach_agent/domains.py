"""Working out which email domain a company uses."""

import re
from urllib.parse import urlparse

# Sites that show up when searching for a company but are never its email domain.
NOT_COMPANY_SITES = (
    "wikipedia.org", "linkedin.com", "crunchbase.com", "glassdoor.com", "indeed.com", "bloomberg.com",
    "forbes.com", "youtube.com", "x.com", "twitter.com", "facebook.com", "instagram.com", "github.com",
    "gitlab.com", "medium.com", "reddit.com", "builtin.com", "wellfound.com", "zoominfo.com", "pitchbook.com",
    "levels.fyi", "yahoo.com", "reuters.com", "cnbc.com", "techcrunch.com",
    # Applicant tracking systems: a job link here says nothing about the company's domain.
    "greenhouse.io", "lever.co", "ashbyhq.com", "myworkdayjobs.com", "workday.com", "smartrecruiters.com",
    "jobvite.com", "icims.com", "bamboohr.com", "workable.com",
)

_SECOND_LEVEL = {"co", "com", "org", "net", "ac"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def base_domain(url_or_host: str) -> str:
    """'https://careers.datadoghq.com/x' -> 'datadoghq.com'; keeps 'company.co.uk' whole."""
    if not url_or_host:
        return ""
    host = urlparse(url_or_host if "//" in url_or_host else f"//{url_or_host}").netloc.lower().split(":")[0]
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_company_site(domain: str) -> bool:
    return bool(domain) and not any(domain == s or domain.endswith("." + s) for s in NOT_COMPANY_SITES)


def rank_domains(company: str, domains: list[str], limit: int = 3) -> list[str]:
    """Dedupe and order candidate domains: exact name match, then containing the name, then the rest.
    'bytedance.com' ranks before 'joinbytedance.com', which ranks before 'other.com'."""
    name = re.sub(r"[^a-z0-9]", "", company.lower())

    def rank(d: str) -> int:
        label = d.split(".")[0].replace("-", "")
        if label == name:
            return 0
        return 1 if name and name in label else 2

    return sorted(dict.fromkeys(d for d in domains if is_company_site(d)), key=rank)[:limit]
