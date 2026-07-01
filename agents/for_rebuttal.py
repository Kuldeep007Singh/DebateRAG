# agents/for_rebuttal.py

from groq import Groq
from typing import List, Dict
from RAG.hybrid_retriever import retrieve_and_rerank
from agents.utils import build_context
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a debate agent arguing FOR the given topic, preparing a rebuttal.
Your opponent just presented their AGAINST argument. Your job is to:
1. Identify the weakest or most questionable claims in their argument
2. Provide counter-evidence that undermines their position
3. Reinforce your original stance with new evidence if possible
4. Be sharp, logical, and avoid emotional language

Rules:
- EVERY claim must be supported by the provided evidence chunks
- Cite paper titles and authors for each point
- Be concise (keep rebuttal to 300-400 words)
- Focus on attacking their logic, not them personally
- If evidence is limited, acknowledge but don't fabricate
"""

def run_for_rebuttal(
    topic: str,
    own_argument: str,
    opposing_argument: str,
    all_chunks: List[Dict]
) -> Dict:
    """
    FOR agent prepares a rebuttal to the AGAINST argument.
    
    Args:
        topic: Original debate topic
        own_argument: FOR agent's opening argument (for context)
        opposing_argument: AGAINST agent's argument (to rebut)
        all_chunks: Available evidence chunks
    
    Returns:
        Dictionary with rebuttal and supporting chunks
    """
    
    # Step 1: Retrieve counter-evidence
    # Construct query that targets weaknesses in the opposing argument
    counter_query = f"criticisms of alternatives to {topic} limitations drawbacks"
    
    chunks = retrieve_and_rerank(
        query=counter_query,
        all_chunks=all_chunks,
        top_k=4,
        stance="FOR"
    )
    
    # Fallback: if no counter-evidence found, use original evidence
    if not chunks:
        print(f"[FOR Rebuttal] No counter-evidence found. Using original evidence.")
        chunks = retrieve_and_rerank(
            query=f"evidence supporting {topic}",
            all_chunks=all_chunks,
            top_k=4,
            stance="FOR"
        )
    
    if not chunks:
        return {
            "role": "FOR_REBUTTAL",
            "rebuttal": "Unable to retrieve evidence for rebuttal. Insufficient sources available.",
            "chunks": [],
            "error": True
        }
    
    context = build_context(chunks)
    
    # Step 2: Construct rebuttal prompt
    user_message = f"""
Topic: {topic}

Your Original FOR Argument:
{own_argument}

Their AGAINST Argument (to rebut):
{opposing_argument}

Counter-Evidence Available:
{context}

Now prepare a sharp rebuttal that:
1. Identifies 1-2 specific weak points in their argument
2. Provides direct counter-evidence (cite sources)
3. Explains why their argument fails
4. Closes with a reinforcement of your position

Keep it to 3-4 paragraphs maximum.
"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        
        rebuttal_text = response.choices[0].message.content
        
        return {
            "role": "FOR_REBUTTAL",
            "rebuttal": rebuttal_text,
            "chunks": chunks
        }
    
    except Exception as e:
        print(f"[FOR Rebuttal] API error: {e}")
        return {
            "role": "FOR_REBUTTAL",
            "rebuttal": f"Error generating rebuttal: {str(e)}",
            "chunks": chunks,
            "error": True
        }