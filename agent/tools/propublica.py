"""Wraps ProPublica's Nonprofit Explorer API, the primary data source for this agent.

Verified directly against the live API before this was written (search.json and
organizations/{ein}.json), not assumed from documentation:
- The search endpoint needs no API key at all, confirmed with a real query.
- The organization endpoint's filings_with_data array does NOT include a
  program-services vs management vs fundraising expense breakdown, only
  totals (totrevenue, totfuncexpns) plus specific named line items
  (compnsatncurrofcr, grsincfndrsng, lessdirfndrsng). A field-by-field
  listing of one real filing (68 fields) confirmed no prgmsrvcs-style key
  exists. That rules out the classic "80% goes to programs" ratio many
  charity-rating sites publish, since this API genuinely does not carry the
  data to compute it, and it is better to compute honest ratios from what is
  actually here than to approximate a number this source cannot support.
"""
from dataclasses import dataclass, field
from typing import Optional

import requests

_SEARCH_URL = "https://projects.propublica.org/nonprofits/api/v2/search.json"
_ORG_URL = "https://projects.propublica.org/nonprofits/api/v2/organizations/{ein}.json"


@dataclass
class OrgMatch:
    ein: int
    name: str
    city: Optional[str]
    state: Optional[str]
    ntee_code: Optional[str]
    score: float


@dataclass
class FilingYear:
    tax_year: int
    total_revenue: Optional[int]
    total_expenses: Optional[int]
    total_assets_end: Optional[int]
    total_liabilities_end: Optional[int]
    total_contributions: Optional[int]
    officer_compensation: Optional[int]
    fundraising_gross_income: Optional[int]
    fundraising_direct_expense: Optional[int]
    pdf_url: Optional[str]
    updated: Optional[str]


@dataclass
class Organization:
    ein: int
    name: str
    city: Optional[str]
    state: Optional[str]
    ntee_code: Optional[str]
    filings: list = field(default_factory=list)  # newest first, list[FilingYear]


def search_charity(name, max_results=5):
    """Searches ProPublica's Nonprofit Explorer by name. No API key required."""
    response = requests.get(_SEARCH_URL, params={"q": name}, timeout=20)
    response.raise_for_status()
    data = response.json()
    matches = []
    for org in data.get("organizations", [])[:max_results]:
        matches.append(OrgMatch(
            ein=org["ein"],
            name=org["name"],
            city=org.get("city"),
            state=org.get("state"),
            ntee_code=org.get("ntee_code"),
            score=org.get("score", 0.0),
        ))
    return matches


def get_organization(ein):
    """Fetches one organization's real filing history by EIN."""
    response = requests.get(_ORG_URL.format(ein=ein), timeout=20)
    response.raise_for_status()
    data = response.json()
    org = data["organization"]

    filings = []
    for f in data.get("filings_with_data", []):
        filings.append(FilingYear(
            tax_year=f.get("tax_prd_yr"),
            total_revenue=f.get("totrevenue"),
            total_expenses=f.get("totfuncexpns"),
            total_assets_end=f.get("totassetsend"),
            total_liabilities_end=f.get("totliabend"),
            total_contributions=f.get("totcntrbgfts"),
            officer_compensation=f.get("compnsatncurrofcr"),
            fundraising_gross_income=f.get("grsincfndrsng"),
            fundraising_direct_expense=f.get("lessdirfndrsng"),
            pdf_url=f.get("pdf_url"),
            updated=f.get("updated"),
        ))
    filings.sort(key=lambda f: f.tax_year or 0, reverse=True)

    return Organization(
        ein=org["ein"],
        name=org["name"],
        city=org.get("city"),
        state=org.get("state"),
        ntee_code=org.get("ntee_code"),
        filings=filings,
    )
