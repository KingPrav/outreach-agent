"""
research_server.py: an MCP server for web research, using the Tavily API.

Tools:
  web_search(query, ...)  -> search results: title, link, snippet, date
  read_page(url)          -> the main text of one web page (e.g. a job posting or blog post)

The agent uses these to learn about each person (talks, blog posts, projects) and
about the role (the job posting, the team, company news) before drafting a message.
Only public, professional information is the goal: nothing personal.

Only this file ever sees your Tavily API key. Results are cached for 7 days in
outputs/.cache/, so repeating a search doesn't spend credits twice.

Normally agent.py starts this server for you. To test it by hand:
    npx @modelcontextprotocol/inspector .venv/bin/python servers/research_server.py
"""

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent.parent          # the outreach-agent folder
load_dotenv(ROOT / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com"
CACHE_DIR = ROOT / "outputs" / ".cache" / "tavily"
CACHE_SECONDS = 7 * 24 * 3600

# LinkedIn pages are left out by default: most content sits behind a login, and
# automated access to LinkedIn is against its terms. The agent can still keep
# the LinkedIn URLs that Hunter returns, so you can open profiles yourself.
DEFAULT_EXCLUDED = ["linkedin.com"]

MAX_PAGE_CHARS = 6000  # keep page text short enough for the model

mcp = MCPServer("research")


# ---------------------------------------------------------------------------
# Helpers (not tools)
# ---------------------------------------------------------------------------
def split_list(text: str) -> list[str]:
    """'a.com, b.com' -> ['a.com', 'b.com']"""
    return [x.strip() for x in text.split(",") if x.strip()]


def cached(kind: str, payload: dict):
    """Return a saved result for this exact request, if it's recent enough."""
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE_DIR / f"{kind}_{key}.json"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_SECONDS:
        return path, json.loads(path.read_text())
    return path, None


async def tavily_post(endpoint: str, body: dict) -> dict:
    """Call a Tavily endpoint. Returns the JSON response, or {"error": "..."}."""
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY is not set. Add it to your .env file."}
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(f"{TAVILY_URL}/{endpoint}", json=body, headers=headers)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, dict):  # Tavily often nests it: {"detail": {"error": "..."}}
            detail = detail.get("error", detail)
        return {"error": f"Tavily API error {response.status_code}: {detail}"}
    return response.json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 5,
    include_domains: str = "",
    exclude_domains: str = "",
    time_range: str = "",
    news: bool = False,
) -> dict:
    """Search the web. Good for finding a person's public professional work
    (conference talks, blog posts, podcasts, open-source projects, interviews)
    and for researching a company or team.

    Args:
        query: What to search for, e.g. '"Jane Doe" Stripe engineering talk'.
            Put a person's name in quotes to avoid results about other people.
        max_results: How many results (1-10).
        include_domains: Comma-separated sites to search only, e.g. "github.com,medium.com".
        exclude_domains: Comma-separated sites to skip (linkedin.com is always skipped).
        time_range: Only recent results: "week", "month" or "year". Empty = any time.
        news: True to search news articles instead of the general web.

    Each result has a title, url, a short snippet of the page, and a date if known.
    """
    body = {
        "query": query,
        "search_depth": "basic",                 # 1 credit ("advanced" costs 2)
        "max_results": max(1, min(int(max_results), 10)),
        "topic": "news" if news else "general",
        "exclude_domains": DEFAULT_EXCLUDED + split_list(exclude_domains),
    }
    if include_domains:
        body["include_domains"] = split_list(include_domains)
    if time_range in {"day", "week", "month", "year"}:
        body["time_range"] = time_range

    path, hit = cached("search", body)
    if hit:
        return {**hit, "from_cache": True}

    data = await tavily_post("search", body)
    if "error" in data:
        return data

    results = [{
        "title": r.get("title"),
        "url": r.get("url"),
        "snippet": (r.get("content") or "")[:500],
        "published_date": r.get("published_date"),
    } for r in data.get("results", [])]
    output = {"query": query, "count": len(results), "results": results}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    return output


@mcp.tool()
async def read_page(url: str) -> dict:
    """Read the main text of one web page, such as a job posting, a blog post,
    or a talk description. Use it on a promising link from web_search.

    Args:
        url: The full page address, starting with https://
    """
    if "linkedin.com" in url:
        return {"error": "LinkedIn pages aren't read automatically. Open this one yourself."}

    body = {"urls": [url], "extract_depth": "basic", "format": "markdown"}
    path, hit = cached("page", body)
    if hit:
        return {**hit, "from_cache": True}

    data = await tavily_post("extract", body)
    if "error" in data:
        return data
    pages = data.get("results", [])
    if not pages:
        failed = data.get("failed_results", [{}])
        return {"error": f"Couldn't read the page: {failed[0].get('error', 'unknown reason')}", "url": url}

    text = pages[0].get("raw_content") or ""
    output = {"url": url, "truncated": len(text) > MAX_PAGE_CHARS, "text": text[:MAX_PAGE_CHARS]}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    # stdio: agent.py starts this file and talks to it over stdin/stdout.
    # Never print() in this file: it would corrupt those messages.
    mcp.run(transport="stdio")
