from agents.orchestrator import run_debate
from RAG.retriever import fetch_papers
from RAG.chunker import chunk_papers
from RAG.vectorstore import embed_and_store, reset_collection

topic = "RAG is better than fine-tuning"
papers = fetch_papers(topic, max_results=10)
chunks = chunk_papers(papers)
reset_collection()
embed_and_store(chunks)

result = run_debate(topic=topic, all_chunks=chunks)

print(f"FOR: {result['for_argument'][:100]}...")
print(f"AGAINST: {result['against_argument'][:100]}...")
print(f"VERDICT: {result['verdict']}")