"""Builds the chat model the LLM-driven nodes (intake, critic, report_writer) use.

api_key is always passed explicitly, never read from a process-wide environment
variable here, because this agent is meant to be deployed as a shared service:
one process can serve many visitors, and baking any one visitor's key into a
process-wide global (an env var, or a cached model instance) risks reusing it
for someone else's query. Every node takes its llm as a parameter rather than
importing this module directly, so tests can substitute a stub chat model with
no network or key involved at all.
"""


def get_chat_model(provider="anthropic", model=None, temperature=0.0, api_key=None):
    if not api_key:
        raise ValueError(f"an API key is required for provider={provider!r}")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-sonnet-4-5-20250929",
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
        )

    raise ValueError(f"unknown provider: {provider!r}, expected 'anthropic' or 'openai'")
