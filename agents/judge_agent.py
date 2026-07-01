# agents/judge_agent.py (IMPROVED)

from groq import Groq
from typing import List, Dict
from agents.utils import call_groq_safe, build_context
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an impartial academic judge evaluating a formal research debate.

Your role:
- Assess BOTH sides fairly based on evidence quality, argument structure, logic, and citation strength
- Do NOT favor either side by default
- Base your verdict ONLY on what was presented, not on your own knowledge
- Be specific about which arguments were stronger and why

Output Format:
You MUST respond with ONLY valid JSON (no other text), structured exactly as follows:
{
  "winner": "FOR" or "AGAINST",
  "confidence": 0.0 to 1.0,
  "summary": "2-3 sentence explanation of why this side won",
  "winner_strengths": ["strength 1", "strength 2", "strength 3"],
  "loser_weaknesses": ["weakness 1", "weakness 2"],
  "evidence_quality": {
    "FOR": 0.0 to 1.0,
    "AGAINST": 0.0 to 1.0
  },
  "argument_structure": {
    "FOR": "well-structured" or "needs improvement",
    "AGAINST": "well-structured" or "needs improvement"
  },
  "citations_strength": {
    "FOR": "strong" or "weak",
    "AGAINST": "strong" or "weak"
  },
  "rebuttal_effectiveness": {
    "FOR": 0.0 to 1.0,
    "AGAINST": 0.0 to 1.0
  },
  "critical_flaws": {
    "FOR": ["flaw1", "flaw2"] or [],
    "AGAINST": ["flaw1", "flaw2"] or []
  }
}
"""


def run_judge_agent(
    topic: str,
    for_argument: str,
    against_argument: str,
    for_chunks: List[Dict],
    against_chunks: List[Dict],
    for_rebuttal: str = None,
    against_rebuttal: str = None
) -> Dict:
    """
    Judge agent evaluates the full debate and returns structured verdict.
    
    Args:
        topic: Debate topic
        for_argument: FOR agent's opening argument
        against_argument: AGAINST agent's opening argument
        for_chunks: Evidence chunks used by FOR agent
        against_chunks: Evidence chunks used by AGAINST agent
        for_rebuttal: FOR agent's rebuttal (optional)
        against_rebuttal: AGAINST agent's rebuttal (optional)
    
    Returns:
        Dictionary with structured verdict
    """
    
    # Build source summaries for judge reference
    for_sources = _summarize_sources(for_chunks) if for_chunks else "No sources used"
    against_sources = _summarize_sources(against_chunks) if against_chunks else "No sources used"
    
    # Construct judge prompt
    prompt = f"""
Topic: {topic}

=== FOR ARGUMENT ===
{for_argument}

Sources Used:
{for_sources}

{"=== FOR REBUTTAL ===" if for_rebuttal else ""}
{for_rebuttal or ""}

=== AGAINST ARGUMENT ===
{against_argument}

Sources Used:
{against_sources}

{"=== AGAINST REBUTTAL ===" if against_rebuttal else ""}
{against_rebuttal or ""}

=== JUDGE INSTRUCTIONS ===
Evaluate this debate holistically:
1. Which side presented stronger evidence?
2. Which side had better argument structure?
3. Which side's citations were more credible?
4. Who won the rebuttal exchanges (if present)?
5. What are the critical flaws in each side?

Declare a winner and explain your reasoning.

RESPOND WITH ONLY THE JSON OBJECT, NO ADDITIONAL TEXT.
"""
    
    try:
        # Call Groq with retry logic
        response_text = call_groq_safe(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            temperature=0.3  # Lower temp for more consistent JSON
        )
        
        # Parse JSON response
        verdict = _parse_judge_response(response_text)
        
        return {
            "role": "JUDGE",
            "verdict": verdict,
            "topic": topic,
            "raw_response": response_text  # Keep for debugging
        }
    
    except Exception as e:
        print(f"[Judge] Error: {e}")
        return {
            "role": "JUDGE",
            "verdict": {
                "error": str(e),
                "winner": "UNKNOWN",
                "confidence": 0.0,
                "summary": "Judge evaluation failed"
            },
            "topic": topic,
            "error": True
        }


def _summarize_sources(chunks: List[Dict]) -> str:
    """Create a brief summary of sources for judge reference."""
    if not chunks:
        return "No sources"
    
    summary = ""
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Unknown")
        authors = metadata.get("authors", "Unknown")
        source = metadata.get("source", "Unknown")
        relevance = chunk.get("relevance_score", 0)
        
        summary += f"{i}. {title} ({authors}) [{source}] - Relevance: {relevance:.2f}\n"
    
    return summary


def _parse_judge_response(response_text: str) -> Dict:
    """
    Parse LLM response into structured JSON.
    Handles various response formats and extracts JSON if embedded in text.
    """
    # Try direct JSON parse first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from text
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: return structured error
    return {
        "error": "Failed to parse judge verdict",
        "winner": "UNKNOWN",
        "confidence": 0.0,
        "summary": "Could not parse judge response",
        "raw_response": response_text
    }


if __name__ == "__main__":
    print("Judge agent loaded. Run via orchestrator.")