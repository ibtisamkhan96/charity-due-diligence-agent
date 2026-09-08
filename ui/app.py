"""Streamlit UI: submits a query to the FastAPI backend, polls for the result, and
shows the agent's step-by-step reasoning log live while it runs.
"""
import os
import time

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Charity Due Diligence Agent", layout="wide")
st.title("Charity Due Diligence Agent")
st.caption(
    "An agent that checks a charity's real IRS filings and recent news before you "
    "donate, and gives an honest verdict, including when the evidence isn't "
    "strong enough to say either way."
)

with st.sidebar:
    st.header("Your API key")
    st.caption(
        "This runs on your own key, not a shared one, so one visitor's usage "
        "can't rate-limit or bill another's. Nothing is stored: the key is sent "
        "with this request only and used for this run."
    )
    provider_label = st.radio(
        "Provider",
        ["Anthropic (Claude) — recommended", "OpenAI"],
        help="Recommended: Anthropic, this agent was built and tested against Claude Sonnet.",
    )
    provider = "anthropic" if provider_label.startswith("Anthropic") else "openai"
    key_help_url = (
        "https://console.anthropic.com/settings/keys"
        if provider == "anthropic"
        else "https://platform.openai.com/api-keys"
    )
    user_api_key = st.text_input(
        f"{'Anthropic' if provider == 'anthropic' else 'OpenAI'} API key",
        type="password",
        placeholder="sk-...",
    )
    st.caption(f"[Get a key]({key_help_url})")

with st.form("query_form"):
    query = st.text_area(
        "Describe the charity you're checking",
        value="Is the American Red Cross a well-run charity?",
        height=80,
    )
    submitted = st.form_submit_button("Check this charity")


def _poll_job(job_id, log_placeholder, timeout_s=120):
    seen_log_lines = 0
    start = time.time()
    while time.time() - start < timeout_s:
        response = requests.get(f"{API_BASE_URL}/query/{job_id}", timeout=30)
        response.raise_for_status()
        status = response.json()

        log_lines = status.get("log", [])
        if len(log_lines) > seen_log_lines:
            log_placeholder.code("\n".join(log_lines), language=None)
            seen_log_lines = len(log_lines)

        if status["status"] in ("completed", "failed"):
            return status
        time.sleep(1.5)
    raise TimeoutError("job did not finish in time")


if submitted and not user_api_key.strip():
    st.warning("Add your API key in the sidebar first, this demo doesn't run on a shared one.")
    st.stop()

if submitted and query.strip():
    with st.spinner("Submitting query..."):
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "provider": provider, "api_key": user_api_key.strip()},
            timeout=30,
        )
        if response.status_code == 400:
            st.error(f"Couldn't start the run: {response.json().get('detail', response.text)}")
            st.stop()
        response.raise_for_status()
        job_id = response.json()["job_id"]

    st.subheader("Agent trace")
    log_placeholder = st.empty()

    try:
        status = _poll_job(job_id, log_placeholder)
    except TimeoutError:
        st.error("The agent didn't finish in time, try again or check the API logs.")
        st.stop()

    if status["status"] == "failed":
        st.error(f"Agent run failed: {status.get('error')}")
        st.stop()

    result = status["result"]

    st.subheader("Report")
    st.markdown(result.get("final_report", "no report produced"))

    health = result.get("financial_health") or {}
    if health:
        st.subheader("Financial snapshot")
        cols = st.columns(4)
        cols[0].metric("Latest filing year", health.get("latest_filing_year", "n/a"))
        surplus = health.get("surplus_ratio")
        cols[1].metric("Surplus ratio", f"{surplus:.1%}" if surplus is not None else "n/a")
        reserve = health.get("reserve_months")
        cols[2].metric("Reserve (months)", f"{reserve:.1f}" if reserve is not None else "n/a")
        exec_share = health.get("exec_comp_share")
        cols[3].metric("Exec. comp. share", f"{exec_share:.1%}" if exec_share is not None else "n/a")
        if health.get("notes"):
            for note in health["notes"]:
                st.info(note)

    news = result.get("news_hits") or []
    news_status = result.get("news_check_status", "skipped")
    with st.expander(f"Recent news ({len(news)})"):
        if news_status == "unavailable":
            st.warning("The news check couldn't be completed (the news source was unreachable or rate-limited). This is not the same as a clean record.")
        elif not news:
            st.markdown("_no recent articles found_")
        for n in news:
            st.markdown(f"- [{n['title']}]({n['url']}) — {n['domain']} ({n['seen_date']})")
