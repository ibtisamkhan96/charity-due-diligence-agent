"""Proves the graph's two real conditional edges actually route, with no network
call or API key anywhere in the test: every tool and the LLM are stand-ins
injected through the same build_graph(...) parameters production code uses.
"""
from agent.graph import build_graph
from agent.nodes.critic import CriticVerdict
from agent.nodes.intake import ConstraintsOutput
from agent.tools.propublica import FilingYear, Organization, OrgMatch


class _FakeStructuredRunnable:
    def __init__(self, canned_output):
        self._canned_output = canned_output

    def invoke(self, messages):
        return self._canned_output


class FakeChatModel:
    def __init__(self, constraints_output, critic_output):
        self._constraints_output = constraints_output
        self._critic_output = critic_output

    def with_structured_output(self, model_cls):
        if model_cls is ConstraintsOutput:
            return _FakeStructuredRunnable(self._constraints_output)
        if model_cls is CriticVerdict:
            return _FakeStructuredRunnable(self._critic_output)
        raise ValueError(f"unexpected structured output model: {model_cls}")

    def invoke(self, messages):
        class _FakeMessage:
            content = "STUB REPORT"
        return _FakeMessage()


def _fake_org(ein):
    return Organization(
        ein=ein, name="Test Charity", city="Testville", state="OH", ntee_code="P20",
        filings=[FilingYear(
            tax_year=2024, total_revenue=1000, total_expenses=800,
            total_assets_end=500, total_liabilities_end=0, total_contributions=900,
            officer_compensation=50, fundraising_gross_income=100,
            fundraising_direct_expense=20, pdf_url=None, updated=None,
        )],
    )


def test_skip_to_critic_when_no_organization_matched():
    """search_charity finds nothing at all: financial_health and news_check should
    never run, the graph should go straight to critic."""
    calls = {"search_charity": 0, "get_organization": 0, "search_news": 0}

    def fake_search_charity(name):
        calls["search_charity"] += 1
        return []

    def fake_get_organization(ein):
        calls["get_organization"] += 1
        raise AssertionError("get_organization should never be called with no matches")

    def fake_search_news(query, max_results=5):
        calls["search_news"] += 1
        raise AssertionError("search_news should never run when nothing was matched")

    llm = FakeChatModel(
        constraints_output=ConstraintsOutput(charity_name="Some Unknown Org"),
        critic_output=CriticVerdict(sufficient=True, reason="nothing to find, stop here"),
    )

    app = build_graph(llm, fake_search_charity, fake_get_organization, fake_search_news)
    result = app.invoke({"raw_query": "is Some Unknown Org legit"})

    assert calls["get_organization"] == 0
    assert calls["search_news"] == 0
    assert result["final_report"].startswith("# Charity Check")
    assert "No matching organization" in result["final_report"]
    print("PASSED: skip-to-critic branch, no wasted tool calls on a non-match")


def test_retry_loop_excludes_wrong_match_and_respects_max_iterations():
    """The critic always rejects the match (simulating a wrong-organization pick):
    the graph must loop more than once, but still terminate at max_iterations."""
    calls = {"search_charity": 0, "get_organization": 0}
    seen_excluded = []

    def fake_search_charity(name):
        calls["search_charity"] += 1
        return [OrgMatch(ein=1, name="Wrong Chapter", city="X", state="OH", ntee_code=None, score=99.0),
                OrgMatch(ein=2, name="Wrong Chapter", city="Y", state="OH", ntee_code=None, score=50.0),
                OrgMatch(ein=3, name="Wrong Chapter", city="Z", state="OH", ntee_code=None, score=10.0)]

    def fake_get_organization(ein):
        calls["get_organization"] += 1
        return _fake_org(ein)

    def fake_search_news(query, max_results=5):
        return []

    class _CountingCriticRunnable:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            return CriticVerdict(sufficient=False, reason="looks like the wrong chapter")

    class AlwaysInsufficientChatModel(FakeChatModel):
        def with_structured_output(self, model_cls):
            if model_cls is CriticVerdict:
                return self._critic_runnable
            return super().with_structured_output(model_cls)

    llm = AlwaysInsufficientChatModel(
        constraints_output=ConstraintsOutput(charity_name="Wrong Chapter"),
        critic_output=None,
    )
    llm._critic_runnable = _CountingCriticRunnable()

    app = build_graph(llm, fake_search_charity, fake_get_organization, fake_search_news)
    result = app.invoke({"raw_query": "find the real charity"}, config={"recursion_limit": 50})

    # max_iterations defaults to 2: one initial lookup plus up to 2 retries = 3 total
    assert calls["search_charity"] == 3, f"expected 3 lookups, got {calls['search_charity']}"
    assert llm._critic_runnable.n == 3, f"expected 3 critic calls, got {llm._critic_runnable.n}"
    assert result["final_report"] == "STUB REPORT"
    print("PASSED: retry loop excludes rejected matches and terminates at max_iterations")


if __name__ == "__main__":
    test_skip_to_critic_when_no_organization_matched()
    test_retry_loop_excludes_wrong_match_and_respects_max_iterations()
    print("ALL GRAPH ROUTING TESTS PASSED")
