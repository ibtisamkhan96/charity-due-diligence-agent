"""Turns a plain-language description into a structured search query.

Genuine language understanding is needed here: people describe charities
loosely ("the red cross", "that local food bank in Ohio"), not by exact legal
name or EIN, so this has to be an LLM call, not keyword matching.
"""
from pydantic import BaseModel, Field

_SYSTEM_PROMPT = (
    "Extract a charity search query from the user's message. Give the clearest "
    "search name to use (drop filler words like 'the' or 'that'), and a US state "
    "abbreviation only if one is clearly mentioned or implied."
)


class ConstraintsOutput(BaseModel):
    charity_name: str = Field(description="the name to search for")
    state_hint: str | None = Field(default=None, description="two-letter US state code, if mentioned")


def make_intake_node(llm):
    structured_llm = llm.with_structured_output(ConstraintsOutput)

    def intake(state):
        result = structured_llm.invoke([
            ("system", _SYSTEM_PROMPT),
            ("human", state["raw_query"]),
        ])
        parsed = result.model_dump()
        return {
            "charity_name": parsed["charity_name"],
            "state_hint": parsed.get("state_hint"),
            "excluded_eins": [],
            "iteration": 0,
            "max_iterations": state.get("max_iterations", 2),
            "log": [f"intake: parsed {state['raw_query']!r} into charity_name={parsed['charity_name']!r}, state_hint={parsed.get('state_hint')!r}"],
        }

    return intake
