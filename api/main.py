"""FastAPI backend wrapping the LangGraph agent.

The LLM key is bring-your-own: each request supplies its own provider and API
key, used to build that one request's own graph and never persisted, logged,
or reused for any other request. Community Cloud/Render/Railway-style shared
deployments serve many callers from one process, and baking any one caller's
key into a module-level global risks reusing it for someone else's query, or
billing this service's owner for traffic that isn't theirs. ProPublica and
GDELT need no server-side secret at all, both are free and keyless. An
optional bearer token (API_AUTH_TOKEN) locks the API itself down when set;
left unset it stays open, the right default for a local demo.
"""
import logging
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import build_graph
from agent.llm import get_chat_model
from agent.tools.news_search import search_news
from agent.tools.propublica import get_organization, search_charity
from api.jobs import JobStore
from api.schemas import JobStatus, QueryRequest, QueryResponse
from observability.logging_config import configure_langsmith, configure_logging

configure_logging()
configure_langsmith()
logger = logging.getLogger("charity_agent.api")

app = FastAPI(
    title="Charity Due Diligence Agent",
    description="Checks a charity's real IRS filings and recent news before you donate.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_job_store = JobStore()


def _build_graph(provider, api_key, workspace_id=None):
    """Not cached: caching by (provider, key) would hold every visitor's key in
    server memory for the process lifetime, and caching without the key would
    reuse whichever caller's key built the graph first for every subsequent
    caller. Graph construction is cheap, in-memory node wiring, no network
    calls, so building it fresh per request costs nothing that matters next to
    the real API calls the agent itself makes."""
    llm = get_chat_model(provider=provider, api_key=api_key, workspace_id=workspace_id)
    return build_graph(llm, search_charity, get_organization, search_news)


def _check_auth(authorization: str = Header(default=None)):
    required_token = os.environ.get("API_AUTH_TOKEN")
    if not required_token:
        return
    if authorization != f"Bearer {required_token}":
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_check_auth)])
def submit_query(request: QueryRequest):
    try:
        graph = _build_graph(request.provider, request.api_key, request.workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job_id = _job_store.create()
    _job_store.run_async(job_id, graph.invoke, {"raw_query": request.query})
    logger.info(f"submitted job {job_id}: {request.query!r}")
    return QueryResponse(job_id=job_id, status="pending")


@app.get("/query/{job_id}", response_model=JobStatus, dependencies=[Depends(_check_auth)])
def get_query(job_id: str):
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return JobStatus(job_id=job_id, **job)
