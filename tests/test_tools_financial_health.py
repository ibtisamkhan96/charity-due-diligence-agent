"""Tests the rule-based financial health calculator against known, hand-checkable
numbers, no network or API key involved."""
from agent.tools.financial_health import compute
from agent.tools.propublica import FilingYear, Organization


def _filing(year, revenue, expenses, assets, liabilities, officer_comp=0,
            fund_gross=0, fund_direct=0):
    return FilingYear(
        tax_year=year, total_revenue=revenue, total_expenses=expenses,
        total_assets_end=assets, total_liabilities_end=liabilities,
        total_contributions=0, officer_compensation=officer_comp,
        fundraising_gross_income=fund_gross, fundraising_direct_expense=fund_direct,
        pdf_url=None, updated=None,
    )


def test_surplus_ratio_and_reserve_months_computed_correctly():
    org = Organization(
        ein=123, name="Test Charity", city="Testville", state="OH", ntee_code="P20",
        filings=[_filing(2024, revenue=1000, expenses=800, assets=2400, liabilities=0)],
    )
    health = compute(org, as_of_year=2025)

    assert health.surplus_ratio == (1000 - 800) / 1000
    assert health.reserve_months == 2400 / (800 / 12)
    assert health.filing_age_years == 1
    assert "old" not in " ".join(health.notes)
    print("PASSED: surplus ratio and reserve months")


def test_stale_filing_flagged():
    org = Organization(
        ein=456, name="Old Data Charity", city="Old Town", state="PA", ntee_code=None,
        filings=[_filing(2019, revenue=500, expenses=500, assets=0, liabilities=0)],
    )
    health = compute(org, as_of_year=2025)

    assert health.filing_age_years == 6
    assert any("years old" in n for n in health.notes)
    print("PASSED: stale filing is flagged")


def test_no_filings_handled_honestly():
    org = Organization(ein=789, name="No Data Charity", city=None, state=None, ntee_code=None, filings=[])
    health = compute(org)

    assert health.years_of_data == 0
    assert health.surplus_ratio is None
    assert "No filings" in health.notes[0]
    print("PASSED: no filings handled without fabricating a number")


def test_fundraising_cost_ratio_and_missing_data_noted():
    org = Organization(
        ein=101, name="No Fundraising Charity", city="X", state="X", ntee_code=None,
        filings=[_filing(2024, revenue=100, expenses=90, assets=50, liabilities=0,
                          fund_gross=0, fund_direct=0)],
    )
    health = compute(org, as_of_year=2025)

    assert health.fundraising_cost_ratio is None
    assert any("fundraising" in n.lower() for n in health.notes)
    print("PASSED: missing fundraising data noted rather than divided by zero")


def test_revenue_trend_computed_across_years():
    org = Organization(
        ein=202, name="Growing Charity", city="X", state="X", ntee_code=None,
        filings=[
            _filing(2024, revenue=1200, expenses=1000, assets=100, liabilities=0),
            _filing(2022, revenue=1000, expenses=900, assets=80, liabilities=0),
        ],
    )
    health = compute(org, as_of_year=2025)

    assert health.revenue_trend_pct == 20.0
    print("PASSED: revenue trend computed across available years")


if __name__ == "__main__":
    test_surplus_ratio_and_reserve_months_computed_correctly()
    test_stale_filing_flagged()
    test_no_filings_handled_honestly()
    test_fundraising_cost_ratio_and_missing_data_noted()
    test_revenue_trend_computed_across_years()
    print("ALL FINANCIAL HEALTH TESTS PASSED")
