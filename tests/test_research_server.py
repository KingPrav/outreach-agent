"""The research server must never search or read LinkedIn or personal social media."""

import importlib.util
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parent.parent / "servers" / "research_server.py"


@pytest.fixture
def server(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("research_server", SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    return module


@pytest.mark.parametrize("url, excluded", [
    ("https://www.instagram.com/p/abc", True),
    ("https://m.facebook.com/someone", True),
    ("https://www.linkedin.com/in/someone", True),
    ("tiktok.com/@someone", True),
    ("https://youtube.com/watch?v=talk", False),
    ("https://x.com/someone/status/1", False),
    ("https://notinstagram.com/post", False),
])
def test_is_excluded(server, url, excluded):
    assert server.is_excluded(url) is excluded


async def test_search_asks_tavily_to_skip_social_and_drops_any_that_slip_through(server, monkeypatch):
    sent = {}

    async def fake_post(endpoint, body):
        sent.update(body)
        return {"results": [
            {"title": "Talk", "url": "https://youtube.com/watch?v=1", "content": "conference talk"},
            {"title": "Photo", "url": "https://www.instagram.com/p/1", "content": "weekend"},
        ]}

    monkeypatch.setattr(server, "tavily_post", fake_post)
    out = await server.web_search('"Jordan Lee" Northwind', exclude_domains="example.com")
    assert {"instagram.com", "facebook.com", "linkedin.com", "example.com"} <= set(sent["exclude_domains"])
    assert [r["url"] for r in out["results"]] == ["https://youtube.com/watch?v=1"]


async def test_read_page_refuses_social_media(server, monkeypatch):
    async def must_not_call(*_):
        raise AssertionError("should not reach Tavily")

    monkeypatch.setattr(server, "tavily_post", must_not_call)
    out = await server.read_page("https://www.instagram.com/p/abc")
    assert "error" in out
