"""
people_server.py: an MCP server that finds people to reach out to, using Hunter.io.

Tools:
  find_people(domain, departments, limit)  -> people at a company, with title, LinkedIn and email
  find_email(full_name, domain)            -> one person's email (for names found elsewhere)
  verify_email(email)                      -> is this address likely to work?
  hunter_credits()                         -> how many Hunter credits you have left

Only this file ever sees your Hunter API key.
Results of find_people are cached for 7 days in outputs/.cache/, so searching the
same company twice doesn't spend credits twice.

Normally agent.py starts this server for you. To test it by hand:
    npx @modelcontextprotocol/inspector .venv/bin/python servers/people_server.py
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent.parent          # the outreach-agent folder
load_dotenv(ROOT / ".env")

HUNTER_API_KEY = os.getenv("HUNTER_API_KEY")
HUNTER_URL = "https://api.hunter.io/v2"
CACHE_DIR = ROOT / "outputs" / ".cache" / "hunter"
CACHE_SECONDS = 7 * 24 * 3600

# httpx logs every request URL, and Hunter's key travels in the URL, so keep those logs quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)

mcp = MCPServer("people")


# ---------------------------------------------------------------------------
# Helpers (not tools): shared by the tools below.
# ---------------------------------------------------------------------------
async def hunter_get(endpoint: str, params: dict) -> dict:
    """Call a Hunter endpoint. Returns {"data": ...} or {"error": "..."}."""
    if not HUNTER_API_KEY:
        return {"error": "HUNTER_API_KEY is not set. Add it to your .env file."}
    params = {k: v for k, v in params.items() if v not in (None, "")}
    params["api_key"] = HUNTER_API_KEY
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{HUNTER_URL}/{endpoint}", params=params)
    body = response.json() if response.content else {}
    if response.status_code == 202:
        return {"pending": True}
    if response.status_code != 200:
        # Hunter errors look like {"errors": [{"details": "..."}]}
        details = "; ".join(e.get("details", "") for e in body.get("errors", [])) or response.text
        return {"error": f"Hunter API error {response.status_code}: {details}"}
    return {"data": body.get("data", {})}


def clean_domain(domain: str) -> str:
    """'https://www.Stripe.com/jobs' -> 'stripe.com'"""
    d = domain.strip().lower().removeprefix("https://").removeprefix("http://").removeprefix("www.")
    return d.split("/")[0]


def cache_path(key: str) -> Path:
    return CACHE_DIR / (key.replace("/", "_").replace(",", "+") + ".json")


# ---------------------------------------------------------------------------
# Tools: what the agent can call.
# ---------------------------------------------------------------------------
@mcp.tool()
async def find_people(domain: str, departments: str = "hr,it,management", limit: int = 20) -> dict:
    """Find people who work at a company, from Hunter.io's database.

    Args:
        domain: The company's website domain, e.g. "stripe.com" (not the company name).
        departments: Comma-separated Hunter departments. Use "hr" for recruiters and
            talent acquisition; "it", "management", "research" or "product" for
            engineering leaders. Default: "hr,it,management".
        limit: Most people to return (1-100). Each 10 people costs 1 Hunter credit.
            The free plan allows at most 10 per search.

    Returns each person's name, position, seniority, department, LinkedIn URL,
    email, and how confident Hunter is about the email.
    """
    domain = clean_domain(domain)
    limit = max(1, min(int(limit), 100))
    key = f"{domain}__{departments}__{limit}"

    # Reuse a recent result instead of spending credits again.
    path = cache_path(key)
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_SECONDS:
        cached = json.loads(path.read_text())
        cached["from_cache"] = True
        return cached

    params = {"domain": domain, "department": departments, "type": "personal", "limit": limit}
    result = await hunter_get("domain-search", params)
    if "error" in result and "limited to 10" in result["error"] and limit > 10:
        # The free plan caps results at 10 per search: retry within the cap.
        params["limit"] = limit = 10
        result = await hunter_get("domain-search", params)
    if "error" in result:
        return result
    data = result["data"]

    people = []
    for e in data.get("emails", []):
        if not (e.get("first_name") and e.get("last_name")):
            continue  # skip entries without a name: nothing personal to write
        people.append({
            "name": f"{e['first_name']} {e['last_name']}",
            "position": e.get("position"),
            "seniority": e.get("seniority"),
            "department": e.get("department"),
            "linkedin": e.get("linkedin"),
            "email": e.get("value"),
            "email_confidence": e.get("confidence"),
            "email_status": (e.get("verification") or {}).get("status"),
        })

    output = {
        "domain": domain,
        "organization": data.get("organization"),
        "email_pattern": data.get("pattern"),   # e.g. "{first}.{last}"
        "count": len(people),
        "people": people,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    return output


@mcp.tool()
async def find_email(full_name: str, domain: str) -> dict:
    """Find the work email of one specific person at a company.
    Use this for someone you found another way (e.g. a name from the company's blog).

    Args:
        full_name: The person's full name, e.g. "Jane Doe".
        domain: The company's domain, e.g. "stripe.com".
    """
    result = await hunter_get("email-finder", {"full_name": full_name, "domain": clean_domain(domain)})
    if "error" in result:
        return result
    d = result["data"]
    return {
        "name": full_name,
        "email": d.get("email"),
        "score": d.get("score"),
        "position": d.get("position"),
        "linkedin": d.get("linkedin_url"),
        "email_status": (d.get("verification") or {}).get("status"),
    }


@mcp.tool()
async def verify_email(email: str) -> dict:
    """Check whether an email address is likely to be delivered, before sending to it.

    Status meanings: "valid" = safe to send; "accept_all" = the company accepts all
    addresses, so it can't be confirmed; "invalid" = don't send; "unknown" = couldn't check.
    """
    for _ in range(3):  # Hunter returns "pending" while it's still checking; ask again
        result = await hunter_get("email-verifier", {"email": email})
        if not result.get("pending"):
            break
        await asyncio.sleep(5)
    if "error" in result:
        return result
    if result.get("pending"):
        return {"email": email, "status": "unknown", "note": "verification still running, try later"}
    d = result["data"]
    return {"email": email, "status": d.get("status"), "score": d.get("score")}


@mcp.tool()
async def hunter_credits() -> dict:
    """How many Hunter searches and verifications are left this month."""
    result = await hunter_get("account", {})
    if "error" in result:
        return result
    requests = result["data"].get("requests", {})
    return {
        "plan": result["data"].get("plan_name"),
        "searches": requests.get("searches"),
        "verifications": requests.get("verifications"),
        "resets_on": result["data"].get("reset_date"),
    }


if __name__ == "__main__":
    # stdio: agent.py starts this file and talks to it over stdin/stdout.
    # Never print() in this file: it would corrupt those messages.
    mcp.run(transport="stdio")
