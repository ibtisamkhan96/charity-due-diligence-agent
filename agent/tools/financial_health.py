"""Computes real financial-health indicators from an Organization's actual filings.

Deliberately rule-based, not an LLM: every number here is arithmetic on real
IRS filing data, and a materials scientist or a donor should be able to
recompute each one by hand from the same filing. No LLM judgment belongs in
this step, only in deciding what the numbers mean afterward (the critic and
report_writer nodes).

Does NOT compute a "percent to programs" ratio: ProPublica's filings_with_data
does not carry the program/management/fundraising expense split (confirmed by
listing every field on a real filing, see agent/tools/propublica.py), so
that popular charity-rating number cannot be honestly produced from this data
source. What follows instead are the ratios this data actually supports.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class FinancialHealth:
    ein: int
    name: str
    latest_filing_year: Optional[int]
    filing_age_years: Optional[int]
    surplus_ratio: Optional[float]           # (revenue - expenses) / revenue, most recent year
    reserve_months: Optional[float]          # (assets - liabilities) / (expenses / 12)
    exec_comp_share: Optional[float]         # officer compensation / total expenses
    fundraising_cost_ratio: Optional[float]  # direct fundraising expense / fundraising gross income
    revenue_trend_pct: Optional[float]       # % change, oldest to newest available filing
    years_of_data: int
    notes: list


def _safe_div(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def compute(org, as_of_year=None):
    """org: agent.tools.propublica.Organization, filings newest first."""
    as_of_year = as_of_year or date.today().year
    notes = []

    if not org.filings:
        return FinancialHealth(
            ein=org.ein, name=org.name, latest_filing_year=None, filing_age_years=None,
            surplus_ratio=None, reserve_months=None, exec_comp_share=None,
            fundraising_cost_ratio=None, revenue_trend_pct=None, years_of_data=0,
            notes=["No filings on record for this EIN."],
        )

    latest = org.filings[0]
    filing_age = (as_of_year - latest.tax_year) if latest.tax_year else None
    if filing_age is not None and filing_age >= 3:
        notes.append(f"Most recent filing is {filing_age} years old, treat any verdict as stale.")

    surplus_ratio = _safe_div(
        (latest.total_revenue - latest.total_expenses) if latest.total_revenue is not None and latest.total_expenses is not None else None,
        latest.total_revenue,
    )

    net_assets = None
    if latest.total_assets_end is not None and latest.total_liabilities_end is not None:
        net_assets = latest.total_assets_end - latest.total_liabilities_end
    reserve_months = _safe_div(net_assets, _safe_div(latest.total_expenses, 12))
    if reserve_months is not None and reserve_months < 0:
        notes.append("Net assets are negative (liabilities exceed assets).")

    exec_comp_share = _safe_div(latest.officer_compensation, latest.total_expenses)

    fundraising_cost_ratio = _safe_div(latest.fundraising_direct_expense, latest.fundraising_gross_income)
    if latest.fundraising_gross_income in (None, 0):
        notes.append("No fundraising income reported this year, fundraising cost ratio not computable.")

    revenue_trend_pct = None
    priced_filings = [f for f in org.filings if f.total_revenue is not None]
    if len(priced_filings) >= 2:
        newest, oldest = priced_filings[0], priced_filings[-1]
        if oldest.total_revenue:
            revenue_trend_pct = ((newest.total_revenue - oldest.total_revenue) / oldest.total_revenue) * 100

    return FinancialHealth(
        ein=org.ein,
        name=org.name,
        latest_filing_year=latest.tax_year,
        filing_age_years=filing_age,
        surplus_ratio=surplus_ratio,
        reserve_months=reserve_months,
        exec_comp_share=exec_comp_share,
        fundraising_cost_ratio=fundraising_cost_ratio,
        revenue_trend_pct=revenue_trend_pct,
        years_of_data=len(org.filings),
        notes=notes,
    )
