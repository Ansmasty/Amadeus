"""
agent/tools/browser.py

Browser automation tools for AMADEUS.
Uses only the stdlib webbrowser module - no external dependencies.
The agent cannot open arbitrary shell commands; only controlled URL navigation.
"""
import re
import webbrowser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from langchain_core.tools import tool


@tool
def open_url(url: str) -> str:
    """
    Open a URL in the system's default web browser.
    Accepts http:// or https:// URLs. If no scheme is provided, https:// is assumed.
    Use this to navigate to any website.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        opened = webbrowser.open(url)
        if opened:
            return f"Opened in browser: {url}"
        return (
            f"Could not open browser (no default browser configured on this system). "
            f"URL was: {url}"
        )
    except Exception as exc:
        return f"Error opening browser: {exc}"


@tool
def search_youtube(query: str) -> str:
    """
    Search YouTube for videos matching the given query.
    Opens the YouTube search results page in the default browser.
    Example: search_youtube("jazz piano tutorial")
    """
    if not query.strip():
        return "Error: Search query cannot be empty."

    encoded = quote_plus(query.strip())
    url = f"https://www.youtube.com/results?search_query={encoded}"
    return open_url.invoke({"url": url})


@tool
def search_web(query: str) -> str:
    """
    Search the web for news, current events, research, and general factual queries.
    Returns a compact summary of top results from DuckDuckGo HTML.
    Use this when the user asks about current events, investigations, or topics that require external sources.
    """
    q = query.strip()
    if not q:
        return "Error: Search query cannot be empty."

    url = "https://html.duckduckgo.com/html/"
    try:
        response = httpx.get(
            url,
            params={"q": q},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
            follow_redirects=True,
        )
        response.raise_for_status()
    except Exception as exc:
        return f"Error searching the web: {exc}"

    html = response.text
    result_blocks = re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', html, re.S)
    snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.S)

    if not result_blocks:
        return f"No se encontraron resultados para: {q}"

    def _strip_tags(text: str) -> str:
        text = re.sub(r"<.*?>", "", text)
        text = text.replace("&quot;", '"').replace("&#39;", "'")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    lines = [f"Resultados web para: {q}"]
    for idx, (href, title_html) in enumerate(result_blocks[:5], start=1):
        title = _strip_tags(title_html)
        parsed_href = urlparse(href)
        query_params = parse_qs(parsed_href.query)
        clean_href = unquote(query_params.get("uddg", [href])[0])
        snippet = _strip_tags(snippets[idx - 1]) if idx - 1 < len(snippets) else ""
        lines.append(f"{idx}. {title}")
        lines.append(f"   {clean_href}")
        if snippet:
            lines.append(f"   {snippet}")

    return "\n".join(lines)
