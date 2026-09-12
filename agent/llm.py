"""Builds the chat model the LLM-driven nodes (intake, critic, report_writer) use.

api_key is always passed explicitly, never read from a process-wide environment
variable here, because this agent is meant to be deployed as a shared service:
one process can serve many visitors, and baking any one visitor's key into a
process-wide global (an env var, or a cached model instance) risks reusing it
for someone else's query. Every node takes its llm as a parameter rather than
importing this module directly, so tests can substitute a stub chat model with
no network or key involved at all.

The Groq *model choice* is a process-wide env var (GROQ_MODEL), unlike the key.
Groq deprecates specific model IDs on a real schedule (22 entries in their own
deprecation log at the time of writing), and this default has already gone
stale twice from under this code within one deploy cycle: openai/gpt-oss-20b
first (works, but hallucinates tool names under with_structured_output, a
documented issue, not fixable here), then llama-3.3-70b-versatile, which
looked current in Groq's own docs but was actually deprecated 2026-08-16 (a
live 404 model_not_found from a real key is what caught it, not the docs). An
env var lets the next deprecation be a Railway variable change, not another
code-diagnose-redeploy cycle.
"""
import os


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
        # qwen/qwen3.6-27b: Groq's own deprecation table names this (or
        # openai/gpt-oss-120b) as the direct replacement for the now-dead
        # llama-3.3-70b-versatile. Not gpt-oss-120b, despite that also being
        # a listed option, because it is the same family as gpt-oss-20b,
        # which is independently documented (langchain-ai/langchain#34155,
        # Groq's own community forum) as unreliable specifically at the
        # with_structured_output() calls this agent depends on. qwen3.6-27b
        # is outside that family and Groq's tool-use docs list it with real
        # structured-output support (json_schema format, constrained
        # decoding). GROQ_MODEL overrides this without a code change, see
        # the module docstring for why that override exists.
        default_model = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
        return ChatGroq(
            model=model or default_model,
            temperature=temperature,
            api_key=api_key,
        )

    raise ValueError(f"unknown provider: {provider!r}, expected 'anthropic', 'openai', or 'groq'")
