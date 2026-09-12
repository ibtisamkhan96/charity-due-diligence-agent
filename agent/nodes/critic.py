"""Decides whether the selected organization and its data are actually usable, or
whether the graph should retry with the next-best matching candidate instead.

This retry exists for a different reason than "gather more options": ProPublica
search can return an unrelated same-named chapter, retirees' association, or
overseas affiliate ahead of the actual organization a donor means (confirmed
live: searching "American Red Cross" surfaces "881458 American Red Cross Club"
and an "Association Of American Red Cross Retirees" in the same result set).
The critic's job is to catch a likely wrong match, not to demand more data from
a correct one. A hard iteration cap still wins regardless of what the critic
would prefer, the same deterministic guardrail used throughout this agent's
sibling project.
"""
from pydantic import BaseModel, Field

_SYSTEM_PROMPT = (
    "You are reviewing whether a charity lookup found the right organization and "
    "enough real data to give a donor an honest verdict, or whether it likely "
    "matched the wrong entity (an unrelated chapter, retirees' association, "
    "employee union, or overseas affiliate sharing the same name) and should "
    "retry with a different candidate. Sufficient means: a real organization was "
    "found with at least one year of financial filing data, and that data "
    "plausibly matches what the user is asking about. If no organization was "
    "found at all, or the matched entity has zero usable financial data, or its "
    "name suggests it is clearly not the entity the user meant (e.g. a retirees' "
    "club when the user asked about the charity itself), say so plainly and mark "
    "it insufficient."
)


class CriticVerdict(BaseModel):
    # A string, not bool: a real Groq run (qwen3.6-27b) generated the tool call
    # with sufficient set to the literal text "True" (Python-style, not JSON
    # true), and Groq's own server-side schema validation rejected the whole
    # request outright for not matching the declared `boolean` type, a 400
    # before this code ever saw a response to coerce. A string field accepts
    # whatever casing a model produces; critic() below is what interprets it,
    # so nothing downstream of this node needs to know the wire type changed.
    sufficient: str = Field(description='Exactly the string "true" or "false".')
    reason: str = Field(description="one or two sentences explaining the call")


def _summarize(state):
    org = state.get("organization") or {}
    health = state.get("financial_health") or {}
    news = state.get("news_hits") or []
    if not org:
        return "No organization was found for this search."
    return (
        f"Matched organization: {org.get('name')} ({org.get('city')}, {org.get('state')}), "
        f"NTEE code {org.get('ntee_code')}. "
        f"Years of filing data: {health.get('years_of_data', 0)}. "
        f"Most recent filing year: {health.get('latest_filing_year')}. "
        f"Notes on data quality: {health.get('notes')}. "
        f"News check status: {state.get('news_check_status', 'skipped')} "
        f"({len(news)} article(s) found)."
    )


def make_critic_node(llm):
    structured_llm = llm.with_structured_output(CriticVerdict)

    def critic(state):
        summary = _summarize(state)
        result = structured_llm.invoke([
            ("system", _SYSTEM_PROMPT),
            ("human", f"Original request: {state['raw_query']}\n\n{summary}"),
        ])
        sufficient = result.sufficient.strip().lower() in ("true", "yes", "1")
        verdict = {"sufficient": sufficient, "reason": result.reason}
        return {
            "critic_verdict": verdict,
            "log": [f"critic: {'sufficient' if sufficient else 'insufficient'}, {verdict['reason']}"],
        }

    return critic


def loop_or_finish(state):
    verdict = state.get("critic_verdict", {})
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)

    if verdict.get("sufficient") or iteration >= max_iterations:
        return "finish"
    return "retry"


def increment_iteration(state):
    excluded = list(state.get("excluded_eins", []))
    if state.get("selected_ein") is not None:
        excluded.append(state["selected_ein"])
    return {
        "iteration": state.get("iteration", 0) + 1,
        "excluded_eins": excluded,
    }
