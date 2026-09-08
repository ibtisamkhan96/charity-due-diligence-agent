"""Wraps the rule-based financial_health calculator as a graph node.

Reconstructs real dataclass instances from the plain-dict state (LangGraph
state has to stay JSON-serialisable) before handing off to compute(), so the
calculation module itself only ever deals with real, typed filing data, not
ambiguous dicts.
"""
from dataclasses import asdict

from agent.tools import financial_health
from agent.tools.propublica import FilingYear, Organization


def financial_health_node(state):
    org_dict = state.get("organization") or {}
    if not org_dict:
        return {
            "financial_health": {},
            "log": ["financial_health: no organization selected, skipped"],
        }

    org = Organization(
        ein=org_dict["ein"],
        name=org_dict["name"],
        city=org_dict.get("city"),
        state=org_dict.get("state"),
        ntee_code=org_dict.get("ntee_code"),
        filings=[FilingYear(**f) for f in org_dict.get("filings", [])],
    )
    health = financial_health.compute(org)
    health_dict = asdict(health)

    return {
        "financial_health": health_dict,
        "log": [f"financial_health: latest filing {health.latest_filing_year}, "
                f"{health.years_of_data} year(s) of data, {len(health.notes)} note(s)"],
    }
