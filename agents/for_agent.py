# agents/for_agent.py

from groq import Groq
from typing import List, Dict
from RAG.hybrid_retriever import retrieve_and_rerank
from agents.utils import build_context, call_groq_safe
import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a professional debate agent arguing FOR the given topic.
Your job is to build a strong, evidence-based argument supporting the topic.
Rules:
- Every claim must reference the provided context chunks
- Cite sources using format: [Source: Paper Title, Authors]
- Be concise, structured, and persuasive
- Do not make up any facts outside the provided context

Example citation:
"RAG improves retrieval accuracy. [Source: Chen et al., 2023: Retrieval Augmented Generation for LLMs]"
"""



def run_for_agent(topic: str, all_chunks: List[Dict]) -> Dict:
    # Now uses hybrid search + reranking instead of plain dense retrieval
    chunks  = retrieve_and_rerank(
        query      = f"evidence supporting {topic}",
        all_chunks = all_chunks,
        top_k      = 5,
        stance = "FOR"
    )
    context = build_context(chunks)

    user_message = f"""
Topic: {topic}

Here is the research evidence you must use to build your argument:

{context}

Now construct a strong FOR argument using only this evidence.
Structure your response as:
1. Opening claim
2. Evidence point 1 (with citation)
3. Evidence point 2 (with citation)
4. Evidence point 3 (with citation)
5. Closing statement
"""
    response = call_groq_safe(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        max_tokens=600
    )

    return {
        "role"     : "FOR",
        "argument" : response,
        "chunks"   : chunks
    }