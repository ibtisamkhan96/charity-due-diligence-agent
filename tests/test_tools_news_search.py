"""Real integration test against GDELT's live, keyless API."""
from agent.tools.news_search import search_news


def test_search_returns_real_articles():
    hits = search_news('"Red Cross"', max_results=5)
    assert len(hits) > 0
    assert all(h.url.startswith("http") for h in hits)
    print(f"PASSED: GDELT returned {len(hits)} real article(s)")


if __name__ == "__main__":
    test_search_returns_real_articles()
    print("ALL NEWS SEARCH INTEGRATION TESTS PASSED")
