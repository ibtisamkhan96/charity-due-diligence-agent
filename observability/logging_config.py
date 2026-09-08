"""Structured logging and LangSmith tracing setup, shared across entry points."""
import logging
import os


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def configure_langsmith():
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "charity-due-diligence-agent")
        logging.getLogger("charity_agent.observability").info(
            f"LangSmith tracing enabled, project={os.environ['LANGCHAIN_PROJECT']}"
        )
    else:
        logging.getLogger("charity_agent.observability").warning(
            "no LANGCHAIN_API_KEY / LANGSMITH_API_KEY set, running without LangSmith "
            "tracing, set one to see the agent's full run trace at smith.langchain.com"
        )
