"""
agent/tools/browser.py

Browser automation tools for AMADEUS.
Uses only the stdlib webbrowser module - no external dependencies.
The agent cannot open arbitrary shell commands; only controlled URL navigation.
"""
import webbrowser
from urllib.parse import quote_plus

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
