# api/main.py

import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents.orchestrator import run_debate
from RAG.retriever import fetch_papers
from RAG.chunker import chunk_papers
from RAG.vectorstore import embed_and_store, reset_collection
from RAG.cache import get_cached_debate, cache_debate, is_redis_available

load_dotenv()

app = FastAPI(
    title   = "DebateRAG API",
    version = "3.0.0"
)


class DebateRequest(BaseModel):
    topic      : str
    max_papers : int = 15
    use_cache  : bool = True


class DebateResponse(BaseModel):
    topic            : str
    for_argument     : str
    against_argument : str
    for_rebuttal     : str
    against_rebuttal : str
    verdict          : str
    cached           : bool = False


# ── Health check ──────────────────────────────────────────
@app.get("/")
async def health_check():
    return {
        "status" : "DebateRAG API v3 is running",
        "redis"  : "connected" if is_redis_available() else "unavailable"
    }


# ── Standard debate endpoint (cached) ────────────────────
@app.post("/debate", response_model=DebateResponse)
async def run_debate_endpoint(request: DebateRequest):
    try:
        # Check cache first
        if request.use_cache and is_redis_available():
            cached = get_cached_debate(request.topic)
            if cached:
                return DebateResponse(**cached, cached=True)

        # Run in thread pool — LangGraph + Groq calls are sync
        result = await asyncio.to_thread(_run_full_debate, request.topic, request.max_papers)

        response_data = {
            "topic"            : result["topic"],
            "for_argument"     : result["for_argument"],
            "against_argument" : result["against_argument"],
            "for_rebuttal"     : result["for_rebuttal"],
            "against_rebuttal" : result["against_rebuttal"],
            "verdict"          : result["verdict"],
        }

        if is_redis_available():
            cache_debate(request.topic, response_data)

        return DebateResponse(**response_data, cached=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming debate endpoint ─────────────────────────────
@app.post("/debate/stream")
async def stream_debate_endpoint(request: DebateRequest):
    """
    Streams debate stages as newline-delimited JSON.
    Each chunk is: {"stage": "...", "content": "..."}
    Stages: for_argument, against_argument, for_rebuttal, against_rebuttal, verdict
    """
    async def event_generator():
        try:
            # Stage 1 — fetch and index papers
            yield _sse("status", "Fetching papers...")
            papers = await asyncio.to_thread(
                fetch_papers, request.topic, request.max_papers
            )

            if not papers:
                yield _sse("error", "No sources found for this topic")
                return

            yield _sse("status", f"Found {len(papers)} sources. Chunking and indexing...")

            # Stage 2 — chunk + embed
            chunks = await asyncio.to_thread(chunk_papers, papers)
            await asyncio.to_thread(reset_collection)
            await asyncio.to_thread(embed_and_store, chunks)

            yield _sse("status", "Running debate agents...")

            # Stage 3 — run each agent and stream results one by one
            from agents.for_agent     import run_for_agent
            from agents.against_agent import run_against_agent
            from agents.judge_agent   import run_judge_agent

            # FOR argument
            yield _sse("status", "FOR agent is building argument...")
            for_result = await asyncio.to_thread(
                run_for_agent, request.topic, chunks
            )
            yield _sse("for_argument", for_result["argument"])

            # AGAINST argument
            yield _sse("status", "AGAINST agent is building argument...")
            against_result = await asyncio.to_thread(
                run_against_agent, request.topic, chunks
            )
            yield _sse("against_argument", against_result["argument"])

            # FOR rebuttal
            yield _sse("status", "FOR agent is preparing rebuttal...")
            for_rebuttal_result = await asyncio.to_thread(
                run_for_agent,
                f"Topic: {request.topic}\nYou already argued FOR.\nNow rebut: {against_result['argument']}",
                chunks
            )
            yield _sse("for_rebuttal", for_rebuttal_result["argument"])

            # AGAINST rebuttal
            yield _sse("status", "AGAINST agent is preparing rebuttal...")
            against_rebuttal_result = await asyncio.to_thread(
                run_against_agent,
                f"Topic: {request.topic}\nYou already argued AGAINST.\nNow rebut: {for_result['argument']}",
                chunks
            )
            yield _sse("against_rebuttal", against_rebuttal_result["argument"])

            # JUDGE verdict
            yield _sse("status", "Judge is deliberating...")
            judge_result = await asyncio.to_thread(
                run_judge_agent,
                topic             = request.topic,
                for_argument      = for_result["argument"],
                against_argument  = against_result["argument"],
                for_rebuttal      = for_rebuttal_result["argument"],
                against_rebuttal  = against_rebuttal_result["argument"]
            )
            yield _sse("verdict", judge_result["verdict"])

            # Cache the full result
            if is_redis_available():
                cache_debate(request.topic, {
                    "topic"            : request.topic,
                    "for_argument"     : for_result["argument"],
                    "against_argument" : against_result["argument"],
                    "for_rebuttal"     : for_rebuttal_result["argument"],
                    "against_rebuttal" : against_rebuttal_result["argument"],
                    "verdict"          : judge_result["verdict"],
                })

            yield _sse("done", "Debate complete")

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse("error", str(e))

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {"X-Accel-Buffering": "no"}
    )


# ── Helpers ───────────────────────────────────────────────
def _sse(stage: str, content: str) -> str:
    """Formats a single SSE line as newline-delimited JSON."""
    return json.dumps({"stage": stage, "content": content}) + "\n"


def _run_full_debate(topic: str, max_papers: int) -> dict:
    """Synchronous full debate — called via asyncio.to_thread."""
    papers = fetch_papers(topic=topic, max_results=max_papers)
    if not papers:
        raise ValueError("No sources found for this topic")
    chunks = chunk_papers(papers)
    reset_collection()
    embed_and_store(chunks)
    return run_debate(topic=topic, all_chunks=chunks)