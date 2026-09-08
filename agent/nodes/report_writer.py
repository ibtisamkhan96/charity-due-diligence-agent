"""Turns the accumulated real data into a final, honest, plain-language report.

Every number handed to the model here was already computed by real API calls
and deterministic arithmetic (financial_health.compute); this node's only job
is to explain what those numbers mean and to say plainly when they are not
enough to support a confident verdict, never to invent a number the graph did
not actually produce.
"""

_SYSTEM_PROMPT = (
    "Write a donor-facing report on the charity described below, using ONLY the "
    "real data provided. Cover: what the organization is, its financial health "
    "(surplus/deficit, reserve months, executive compensation share, fundraising "
    "cost ratio, revenue trend), how recent the underlying filing is, and the "
    "news check result. If the news check status is 'unavailable', say plainly "
    "that no news check could be completed, do not imply a clean record when "
    "none was actually confirmed. If the filing is old, data is missing, or the "
    "match looks wrong, say so plainly too rather than giving false confidence. "
    "Never state a number that was not given to you. End with one clear verdict "
    "sentence: recommended, recommended with caveats, or cannot confirm."
)


def make_report_writer_node(llm):
    def report_writer(state):
        org = state.get("organization") or {}
        health = state.get("financial_health") or {}
        news = state.get("news_hits") or []
        verdict = state.get("critic_verdict") or {}

        if not org:
            report = (
                f"# Charity Check: {state.get('charity_name', state['raw_query'])}\n\n"
                "No matching organization could be found in ProPublica's Nonprofit "
                "Explorer database after checking available candidates. "
                "**Cannot confirm.** Verify the exact legal name and try again, "
                "or check the organization's own website for its EIN."
            )
            return {"final_report": report, "log": ["report_writer: no organization found, wrote a cannot-confirm report"]}

        news_status = state.get("news_check_status", "skipped")
        context = (
            f"Organization: {org.get('name')} ({org.get('city')}, {org.get('state')}), EIN {org.get('ein')}\n"
            f"Financial health: {health}\n"
            f"News check status: {news_status}\n"
            f"Recent news ({len(news)} articles): {news}\n"
            f"Critic's assessment: {verdict}\n"
        )
        message = llm.invoke([
            ("system", _SYSTEM_PROMPT),
            ("human", context),
        ])
        report = message.content
        return {"final_report": report, "log": ["report_writer: composed final report"]}

    return report_writer
