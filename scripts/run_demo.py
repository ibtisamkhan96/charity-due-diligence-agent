"""Runs the agent end to end from the command line, no API/UI involved, meant for
recording a demo video or a quick sanity check with a real key.

Usage:
    python scripts/run_demo.py "Is the American Red Cross a well-run charity?"

Reads ANTHROPIC_API_KEY (or OPENAI_API_KEY with LLM_PROVIDER=openai) from the
environment, since agent.llm.get_chat_model requires an explicit key and never
falls back to a process-wide env var itself, that restriction is deliberate for
the deployed, multi-visitor API, but this script is a single local run, so it's
fine to read the key here and pass it in explicitly.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from agent.graph import build_graph
from agent.llm import get_chat_model
from agent.tools.news_search import search_news
from agent.tools.propublica import get_organization, search_charity
from observability.logging_config import configure_langsmith, configure_logging


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    query = sys.argv[1]

    configure_logging()
    configure_langsmith()

    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    api_key = os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic" else os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(f"Set {'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'} first.")

    llm = get_chat_model(provider=provider, api_key=api_key)
    app = build_graph(llm, search_charity, get_organization, search_news)

    print(f"\nRunning: {query}\n" + "-" * 60)
    result = app.invoke({"raw_query": query}, config={"recursion_limit": 50})

    print("\n" + "=" * 60)
    print("AGENT TRACE")
    print("=" * 60)
    for line in result.get("log", []):
        print(f"  - {line}")

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.get("final_report", "no report produced"))


if __name__ == "__main__":
    main()
