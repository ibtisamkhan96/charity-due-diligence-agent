"""Searches ProPublica for the named charity and fetches its real filing history.

On a retry, the previously selected EIN is excluded so the graph tries the
next-best match instead of re-fetching the same wrong organization, the same
"don't repeat a rejected candidate" logic as the substitution retry in the
battery-electrode-screening-agent.
"""
from dataclasses import asdict


def make_charity_lookup_node(search_fn, get_org_fn):
    """search_fn matches propublica.search_charity's signature, get_org_fn matches
    propublica.get_organization's signature, both injected for testability."""

    def charity_lookup(state):
        excluded = set(state.get("excluded_eins", []))
        matches = search_fn(state["charity_name"])

        state_hint = state.get("state_hint")
        candidates = [m for m in matches if m.ein not in excluded]
        if state_hint:
            in_state = [m for m in candidates if m.state == state_hint]
            if in_state:
                candidates = in_state

        if not candidates:
            return {
                "candidates": [],
                "selected_ein": None,
                "organization": {},
                "log": [f"charity_lookup: no remaining match for {state['charity_name']!r}"],
            }

        best = max(candidates, key=lambda m: m.score)
        org = get_org_fn(best.ein)

        return {
            "candidates": [asdict(m) for m in candidates],
            "selected_ein": best.ein,
            "organization": {
                "ein": org.ein,
                "name": org.name,
                "city": org.city,
                "state": org.state,
                "ntee_code": org.ntee_code,
                "filings": [asdict(f) for f in org.filings],
            },
            "log": [f"charity_lookup: selected EIN {best.ein} ({org.name}, {org.city}, {org.state}) from {len(candidates)} candidate(s)"],
        }

    return charity_lookup
