"""News grounding via GDELT's DOC 2.0 API.

Chosen over Google News' RSS feed and over NewsAPI.org: Google News RSS's own
copyright notice explicitly restricts it to "personal, non-commercial use...
within a personal feed reader," which a deployed public tool would violate.
NewsAPI.org's free tier is documented for development/testing only, not
production. GDELT is a research project explicitly built for this kind of
programmatic, large-scale news access, free, and keyless, confirmed live
before this was written.

GDELT asks callers directly (in its own error response) to keep requests to
one per 5 seconds. _last_call_at enforces that regardless of how many times
this module is called within one agent run, rather than trusting every
caller to pace itself. That in-process throttle only knows about requests
this process made, though, not the server's actual cooldown window, so a
429 still surfaced in real testing shortly after unrelated manual requests
from the same machine. retry_with_backoff (the same pattern already proven
against arXiv's rate limit) covers that gap.
"""
import time
from dataclasses import dataclass
from functools import wraps

import requests

_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_MIN_INTERVAL_S = 6.0
_last_call_at = 0.0


def retry_with_backoff(max_attempts=3, base_delay=5.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2
        return wrapper
    return decorator


@dataclass
class NewsHit:
    title: str
    url: str
    domain: str
    seen_date: str
    source_country: str


def _throttle():
    global _last_call_at
    elapsed = time.time() - _last_call_at
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _last_call_at = time.time()


@retry_with_backoff(max_attempts=3, base_delay=6.0)
def _run_query(query, max_results):
    _throttle()
    response = requests.get(
        _DOC_API,
        params={
            "query": query,
            "mode": "artlist",
            "maxrecords": max_results,
            "format": "json",
            "sort": "datedesc",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def search_news(query, max_results=5):
    """Searches GDELT for recent articles mentioning `query`. No API key needed."""
    data = _run_query(query, max_results)

    hits = []
    for a in data.get("articles", [])[:max_results]:
        hits.append(NewsHit(
            title=a.get("title", ""),
            url=a.get("url", ""),
            domain=a.get("domain", ""),
            seen_date=a.get("seendate", ""),
            source_country=a.get("sourcecountry", ""),
        ))
    return hits
