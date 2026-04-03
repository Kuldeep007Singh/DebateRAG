from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agents.orchestrator import run_debate
from RAG.retriever import fetch_papers
from RAG.chunker import chunk_papers
from RAG.vectorstore import embed_and_store, reset_collection

load_dotenv()

app = FastAPI(
    title   = "DebateRAG API",
    version = "1.0.0"
)


# ── Request / Response schemas ────────────────────────────
class DebateRequest(BaseModel):
    topic      : str
    max_papers : int = 15


class DebateResponse(BaseModel):
    topic            : str
    for_argument     : str
    against_argument : str
    for_rebuttal     : str
    against_rebuttal : str
    verdict          : str


# ── Health check ──────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "DebateRAG API is running"}


# ── Main debate endpoint ──────────────────────────────────
@app.post("/debate", response_model=DebateResponse)
def run_debate_endpoint(request: DebateRequest):
    try:
        # Step 1 — fetch and index papers
        papers = fetch_papers(
            topic       = request.topic,
            max_results = request.max_papers
        )
        if not papers:
            raise HTTPException(
                status_code = 404,
                detail      = "No papers found for this topic"
            )

        chunks = chunk_papers(papers)
        reset_collection()
        embed_and_store(chunks)

        # Step 2 — run the full debate
        result = run_debate(request.topic)

        # Step 3 — return structured response
        return DebateResponse(
            topic            = result["topic"],
            for_argument     = result["for_argument"],
            against_argument = result["against_argument"],
            for_rebuttal     = result["for_rebuttal"],
            against_rebuttal = result["against_rebuttal"],
            verdict          = result["verdict"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
