# RAG/query_rewriter.py

from groq import Groq
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rewrite_query(topic: str, stance: str) -> List[str]:
    """
    Takes a debate topic and stance (FOR/AGAINST),
    returns 3 optimized search queries for better retrieval.
    """
    prompt = f"""
You are a search query optimization expert.
Given a debate topic and a stance, generate exactly 3 distinct search queries
that would retrieve the most relevant evidence for that stance.

Topic: {topic}
Stance: {stance}

Rules:
- Each query must be different — cover different angles
- Queries should be specific and information-dense
- No numbering, no bullets, no explanation
- Return exactly 3 queries, one per line, nothing else
"""

    response = client.chat.completions.create(
        model      = "llama-3.3-70b-versatile",
        max_tokens = 200,
        messages   = [{"role": "user", "content": prompt}]
    )

    raw      = response.choices[0].message.content.strip()
    queries  = [q.strip() for q in raw.split("\n") if q.strip()]
    return queries[:3]  # safety cap at 3


if __name__ == "__main__":
    queries = rewrite_query(
        topic  = "RAG is better than fine-tuning for enterprise LLMs",
        stance = "FOR"
    )
    for q in queries:
        print(q)