# api/main.py

from fastapi import FastAPI, HTTPException
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
    version = "2.0.0"
)


class DebateRequest(BaseModel):
    topic      : str
    max_papers : int = 15
    use_cache  : bool = True   # ← caller can bypass cache if needed


class DebateResponse(BaseModel):
    topic            : str
    for_argument     : str
    against_argument : str
    for_rebuttal     : str
    against_rebuttal : str
    verdict          : str
    cached           : bool = False   # ← tells caller if result came from cache


@app.get("/")
def health_check():
    return {
        "status" : "DebateRAG API v2 is running",
        "redis"  : "connected" if is_redis_available() else "unavailable"
    }


@app.post("/debate", response_model=DebateResponse)
def run_debate_endpoint(request: DebateRequest):
    try:
        # Step 1 — check cache first
        if request.use_cache and is_redis_available():
            cached = get_cached_debate(request.topic)
            if cached:
                return DebateResponse(**cached, cached=True)

        # Step 2 — fetch papers (arxiv + wikipedia fallback)
        papers = fetch_papers(
            topic       = request.topic,
            max_results = request.max_papers
        )
        if not papers:
            raise HTTPException(
                status_code = 404,
                detail      = "No sources found for this topic"
            )

        # Step 3 — chunk + store in ChromaDB
        chunks = chunk_papers(papers)
        reset_collection()
        embed_and_store(chunks)

        # Step 4 — run debate
        result = run_debate(
            topic      = request.topic,
            all_chunks = chunks
        )

        # Step 5 — build response dict
        response_data = {
            "topic"            : result["topic"],
            "for_argument"     : result["for_argument"],
            "against_argument" : result["against_argument"],
            "for_rebuttal"     : result["for_rebuttal"],
            "against_rebuttal" : result["against_rebuttal"],
            "verdict"          : result["verdict"],
        }

        # Step 6 — store in cache
        if is_redis_available():
            cache_debate(request.topic, response_data)

        return DebateResponse(**response_data, cached=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)