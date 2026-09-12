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


def get_chat_model(provider="anthropic", model=None, temperature=0.0, api_key=None, workspace_id=None):
    if not api_key:
        raise ValueError(f"an API key is required for provider={provider!r}")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Real 400 from a live key: "This API key is not scoped to a workspace,
        # so this request must include the anthropic-workspace-id header".
        # Anthropic's newer identity-linked personal/service-account keys can
        # reach more than one workspace, and a request from one of those must
        # say which workspace to run in, there is no way to omit this for that
        # key type. A single-workspace key never needs it, so this stays
        # optional and does nothing for the common case.
        extra_kwargs = {}
        if workspace_id:
            extra_kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
        return ChatAnthropic(
            model=model or "claude-sonnet-4-5-20250929",
            temperature=temperature,
            api_key=api_key,
            **extra_kwargs,
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
        actual_model = model or default_model

        extra_kwargs = {}
        if "qwen" in actual_model.lower():
            # qwen3.6-27b defaults to "thinking" mode, which spends hidden
            # reasoning tokens on every call, including for a short structured
            # extraction or a short donor report, neither of which needs "complex
            # reasoning, math, or coding" (Groq's own description of when
            # thinking mode is for). That bloat is what a real 429 caught:
            # requested 1780 output tokens against the free tier's 1000/minute
            # cap. reasoning_effort is a real ChatGroq constructor field in the
            # installed langchain-groq version (verified directly, not passed
            # via model_kwargs, which this version's own pydantic validation
            # rejects for any parameter it recognises explicitly). Scoped to
            # qwen models specifically, since a different model reached via
            # GROQ_MODEL might not support it the same way.
            extra_kwargs["reasoning_effort"] = "none"

        return ChatGroq(
            model=actual_model,
            temperature=temperature,
            api_key=api_key,
            **extra_kwargs,
        )

    raise ValueError(f"unknown provider: {provider!r}, expected 'anthropic', 'openai', or 'groq'")
