# api/main.py (IMPROVED)

import asyncio
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents.orchestrator import run_debate
from RAG.retriever import fetch_papers
from RAG.chunker import chunk_papers
from RAG.vectorstore import embed_and_store, reset_collection
from RAG.cache import get_cached_debate, cache_debate, is_redis_available

from fastapi.middleware.cors import CORSMiddleware



load_dotenv()

app = FastAPI(
    title="DebateRAG API",
    version="4.0.0",
    description="Multi-agent RAG debate system with evidence-based arguments"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace * with your Streamlit URL
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Request/Response Models ────────────────────────────────────
class DebateRequest(BaseModel):
    """Validated debate request."""
    topic: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="The debate topic (5-200 characters)"
    )
    max_papers: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Number of papers to retrieve (5-50)"
    )
    use_cache: bool = Field(
        default=True,
        description="Use cached result if available"
    )


class DebateResponse(BaseModel):
    """Full debate response."""
    topic: str
    for_argument: str
    against_argument: str
    for_rebuttal: str
    against_rebuttal: str
    verdict: dict
    cached: bool = False
    generation_time_seconds: float = 0.0


# ── Health & Info ──────────────────────────────────────────────
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "DebateRAG API v4 is running",
        "redis": "connected" if is_redis_available() else "unavailable",
        "timestamp": time.time()
    }


@app.get("/health")
async def detailed_health():
    """Detailed system health."""
    return {
        "api_version": "4.0.0",
        "redis_available": is_redis_available(),
        "uptime_seconds": time.time(),
        "endpoints": [
            "/debate (POST) - Standard debate (cached)",
            "/debate/stream (POST) - Streaming debate",
            "/health (GET) - This endpoint"
        ]
    }


# ── Standard Debate Endpoint (Cached) ──────────────────────────
@app.post("/debate", response_model=DebateResponse)
async def run_debate_endpoint(request: DebateRequest):
    """
    Run a full debate and cache the result.
    Returns immediately from cache if available.
    
    Args:
        request: DebateRequest with topic and options
    
    Returns:
        DebateResponse with all debate stages
    
    Raises:
        HTTPException 500 if debate generation fails
    """
    start_time = time.time()
    
    try:
        # Check cache first
        if request.use_cache and is_redis_available():
            cached_result = get_cached_debate(request.topic)
            if cached_result:
                return DebateResponse(
                    **cached_result,
                    cached=True,
                    generation_time_seconds=time.time() - start_time
                )
        
        # Run in thread pool (sync operations)
        result = await asyncio.to_thread(
            _run_full_debate,
            request.topic,
            request.max_papers
        )
        
        response_data = DebateResponse(
            **result,
            cached=False,
            generation_time_seconds=time.time() - start_time
        )
        
        # Cache the result
        if is_redis_available():
            try:
                cache_debate(request.topic, result)
            except Exception as e:
                print(f"[Cache] Non-blocking error: {e}")
        
        return response_data

    except ValueError as e:
        # Specific error (no sources found, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Debate generation failed: {str(e)}")


# ── Streaming Debate Endpoint ──────────────────────────────────
@app.post("/debate/stream")
async def stream_debate_endpoint(request: DebateRequest):
    """
    Stream debate stages as newline-delimited JSON.
    Each line is: {"stage": "...", "content": "...", "timestamp": float}
    
    Stages:
    - status: Progress updates
    - heartbeat: Keep-alive messages
    - for_argument, against_argument: Opening arguments
    - for_rebuttal, against_rebuttal: Rebuttal stages
    - verdict: Final judge verdict
    - done: Completion signal
    - error: Error message
    """
    async def event_generator():
        try:
            # Stage 1: Fetch papers
            yield _sse("status", "Fetching papers from arXiv and Wikipedia...", "paper_fetch_start")
            
            try:
                papers = await asyncio.to_thread(
                    fetch_papers,
                    request.topic,
                    request.max_papers
                )
            except Exception as e:
                yield _sse("error", f"Paper fetch failed: {str(e)}", "paper_fetch_error")
                return
            
            if not papers:
                yield _sse("error", f"No sources found for topic: {request.topic}", "no_papers")
                return
            
            yield _sse("status", f"Found {len(papers)} sources. Processing...", "papers_found")
            
            # Stage 2: Chunk and embed
            try:
                chunks = await asyncio.to_thread(chunk_papers, papers)
                await asyncio.to_thread(reset_collection)
                await asyncio.to_thread(embed_and_store, chunks)
                
                yield _sse(
                    "status",
                    f"Indexed {len(chunks)} chunks. Starting debate...",
                    "indexing_complete"
                )
            except Exception as e:
                yield _sse("error", f"Indexing failed: {str(e)}", "indexing_error")
                return
            
            # Stage 3: Run orchestrator (which internally manages all agents)
            try:
                yield _sse("status", "FOR agent building argument...", "for_agent_start")
                
                debate_result = await asyncio.to_thread(
                    run_debate,
                    request.topic,
                    chunks,
                    request.max_papers
                )
                
                # Stream each stage of the result
                for stage in ["for_argument", "against_argument", "for_rebuttal", "against_rebuttal", "verdict"]:
                    content = debate_result.get(stage, "")
                    if content:
                        yield _sse(stage, str(content), f"{stage}_complete")
            
            except Exception as e:
                yield _sse("error", f"Debate generation failed: {str(e)}", "debate_error")
                return
            
            # Cache the result
            if is_redis_available():
                try:
                    cache_debate(request.topic, debate_result)
                except Exception as e:
                    print(f"[Cache] Non-blocking error: {e}")
            
            yield _sse("done", "Debate complete", "debate_complete")

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse("error", f"Unexpected error: {str(e)}", "unexpected_error")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )


# ── Helper Functions ──────────────────────────────────────────
def _sse(stage: str, content: str, event_id: str = None) -> str:
    """
    Format Server-Sent Event (newline-delimited JSON).
    
    Args:
        stage: Event stage name
        content: Event content
        event_id: Optional unique event ID
    
    Returns:
        JSON string with trailing newline
    """
    payload = {
        "stage": stage,
        "content": content,
        "timestamp": time.time()
    }
    if event_id:
        payload["event_id"] = event_id
    
    return json.dumps(payload) + "\n"


def _run_full_debate(topic: str, max_papers: int) -> dict:
    """
    Synchronous wrapper for full debate.
    Called via asyncio.to_thread from async context.
    
    Args:
        topic: Debate topic
        max_papers: Max papers to retrieve
    
    Returns:
        Dictionary with debate stages and results
    
    Raises:
        ValueError if no papers found
        Exception if any stage fails
    """
    # Fetch and prepare papers
    papers = fetch_papers(topic=topic, max_results=max_papers)
    if not papers:
        raise ValueError(f"No sources found for topic: {topic}")
    
    # Chunk and embed
    chunks = chunk_papers(papers)
    reset_collection()
    embed_and_store(chunks)
    
    # Run orchestrated debate
    return run_debate(topic=topic, all_chunks=chunks, max_papers=max_papers)


# ── Error Handlers ─────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )