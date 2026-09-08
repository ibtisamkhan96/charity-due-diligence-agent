"""Wires the state graph together.

Two real conditional edges:

  charity_lookup --[no organization matched]--> critic (skip financial/news checks
                 --[matched something]--> financial_health_node -> news_check -> critic
                    entirely, nothing to check)

  critic --[sufficient, or out of retries]--> report_writer -> END
         --[insufficient, retries left]--> increment_iteration -> charity_lookup (loop)

The second edge is the actual cycle: a wrong-organization match (a same-named
chapter or retirees' association outranking the real charity in search results)
gets excluded and the graph tries the next-best candidate, bounded by a hard
iteration cap so a model that keeps rejecting matches can never loop forever.
"""
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.intake import make_intake_node
from agent.nodes.charity_lookup import make_charity_lookup_node
from agent.nodes.financial_health_node import financial_health_node
from agent.nodes.news_check import make_news_check_node
from agent.nodes.critic import make_critic_node, loop_or_finish, increment_iteration
from agent.nodes.report_writer import make_report_writer_node


def needs_financial_check(state):
    return "check" if state.get("selected_ein") is not None else "skip_to_critic"


def build_graph(llm, search_charity_fn, get_organization_fn, search_news_fn):
    """Every tool and the LLM are injected as parameters, not imported inside node
    modules and called directly, so this same function builds a fully real graph
    in production and a fully mocked one in tests."""
    graph = StateGraph(AgentState)

    graph.add_node("intake", make_intake_node(llm))
    graph.add_node("charity_lookup", make_charity_lookup_node(search_charity_fn, get_organization_fn))
    graph.add_node("financial_health", financial_health_node)
    graph.add_node("news_check", make_news_check_node(search_news_fn))
    graph.add_node("critic", make_critic_node(llm))
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("report_writer", make_report_writer_node(llm))

    graph.set_entry_point("intake")
    graph.add_edge("intake", "charity_lookup")
    graph.add_conditional_edges("charity_lookup", needs_financial_check, {
        "check": "financial_health",
        "skip_to_critic": "critic",
    })
    graph.add_edge("financial_health", "news_check")
    graph.add_edge("news_check", "critic")
    graph.add_conditional_edges("critic", loop_or_finish, {
        "retry": "increment_iteration",
        "finish": "report_writer",
    })
    graph.add_edge("increment_iteration", "charity_lookup")
    graph.add_edge("report_writer", END)

    return graph.compile()
