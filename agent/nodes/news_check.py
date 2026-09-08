"""Checks for recent news about the selected organization, catching controversies
or investigations that would not yet show up in a tax filing, which typically
lags a year or two behind the organization's actual current situation.
"""
import logging
from dataclasses import asdict

logger = logging.getLogger("charity_agent.news_check")


def make_news_check_node(search_fn):
    """search_fn matches agent.tools.news_search.search_news's signature."""

    def news_check(state):
        org = state.get("organization") or {}
        name = org.get("name")
        if not name:
            return {"news_hits": [], "news_check_status": "skipped",
                    "log": ["news_check: no organization selected, skipped"]}

        try:
            hits = search_fn(f'"{name}" investigation OR controversy OR fraud')
        except Exception as e:
            # A rate-limited or unreachable news source is a reason to report
            # without a news check, not a reason to fail the whole run: the
            # financial filing data still stands on its own. news_check_status
            # keeps "couldn't check" distinct from "checked, found nothing",
            # so the report can say plainly which one happened rather than
            # implying a clean bill of health it never actually confirmed.
            logger.warning(f"news search for {name!r} failed, continuing without a news check: {e}")
            return {"news_hits": [], "news_check_status": "unavailable",
                    "log": [f"news_check: unavailable ({e.__class__.__name__}), continuing without it"]}

        return {
            "news_hits": [asdict(h) for h in hits],
            "news_check_status": "ok",
            "log": [f"news_check: {len(hits)} recent article(s) for {name!r}"],
        }

    return news_check
