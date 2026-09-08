"""Real integration test against ProPublica's live, keyless API, confirming the
actual field names this project depends on still exist and behave as expected."""
from agent.tools.propublica import get_organization, search_charity


def test_search_finds_a_real_well_known_charity():
    matches = search_charity("American Red Cross")
    assert len(matches) > 0
    assert any("Red Cross" in m.name for m in matches)
    print(f"PASSED: search returned {len(matches)} real match(es)")


def test_get_organization_returns_real_filing_data():
    org = get_organization(530196605)  # American National Red Cross, confirmed EIN
    assert org.name == "American National Red Cross"
    assert len(org.filings) > 0
    latest = org.filings[0]
    assert latest.total_revenue is not None and latest.total_revenue > 0
    assert latest.total_expenses is not None and latest.total_expenses > 0
    print(f"PASSED: fetched {len(org.filings)} real filing year(s) for {org.name}")


if __name__ == "__main__":
    test_search_finds_a_real_well_known_charity()
    test_get_organization_returns_real_filing_data()
    print("ALL PROPUBLICA INTEGRATION TESTS PASSED")
