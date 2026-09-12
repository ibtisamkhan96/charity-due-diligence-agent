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

    if provider == "groq":
        from langchain_groq import ChatGroq
        # NOT openai/gpt-oss-20b: this project's only real Groq usage is
        # with_structured_output() (intake, critic), and gpt-oss-20b/120b are
        # documented as unreliable there, both in a filed LangChain issue
        # (langchain-ai/langchain#34155) and on Groq's own community forum,
        # hallucinating a slightly wrong tool name and getting the whole
        # request rejected server-side (tool_use_failed). llama-3.3-70b-versatile
        # is the model that same issue confirmed works correctly with both
        # tool-calling strategies, and is still a current, active production
        # model on Groq as of this writing, not the deprecated alias
        # (llama3-70b-8192, no "3.3"/"-versatile") an earlier version of this
        # comment mistook it for.
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=api_key,
        )

    raise ValueError(f"unknown provider: {provider!r}, expected 'anthropic', 'openai', or 'groq'")
