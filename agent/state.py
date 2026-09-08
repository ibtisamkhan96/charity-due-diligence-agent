"""The shared state every node reads from and writes to.

log accumulates across loop iterations (operator.add) rather than being
overwritten each round, so the full trace survives a retry, the same reason
this pattern is used in the battery-electrode-screening-agent: a reducer is
what lets LangGraph merge a node's partial update into the running total
instead of replacing it.
"""
import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict, total=False):
    raw_query: str
    charity_name: str
    state_hint: Optional[str]

    candidates: list          # list[dict], from the most recent lookup
    excluded_eins: list       # EINs already tried, so a retry picks a different match
    selected_ein: Optional[int]
    organization: dict        # the selected org's real filing data, as a dict

    financial_health: dict
    news_hits: list
    news_check_status: str  # "ok" | "unavailable" | "skipped"

    iteration: int
    max_iterations: int
    critic_verdict: dict

    final_report: str
    log: Annotated[list, operator.add]
