from agents.for_rebuttal import run_for_rebuttal
from RAG.retriever import fetch_papers
from RAG.chunker import chunk_papers

papers = fetch_papers("RAG vs fine-tuning", max_results=10)
chunks = chunk_papers(papers)

result = run_for_rebuttal(
    topic="RAG is better than fine-tuning",
    own_argument="RAG improves accuracy without retraining...",
    opposing_argument="Fine-tuning provides better context understanding...",
    all_chunks=chunks
)

print(result["rebuttal"])