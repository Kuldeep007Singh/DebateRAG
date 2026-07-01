# agents/utils.py

from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_context(chunks: List[Dict]) -> str:
    """
    Format chunks into a readable context string for agents.
    Standardized format used by all agents.
    """
    if not chunks:
        return "[No evidence available]"
    
    context = ""
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "arxiv")
        relevance = chunk.get("relevance_score", "N/A")
        
        context += f"""
[{i}] Source  : {source}
    Title   : {metadata.get('title', 'Unknown')}
    Authors : {metadata.get('authors', 'Unknown')}
    URL     : {metadata.get('url', '')}
    Relevance : {relevance}
    
    {chunk.get('text', 'No text available')}
    
---"""
    return context


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_groq_safe(
    messages: List[Dict],
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 600,
    temperature: float = 0.7
) -> str:
    """
    Safe Groq API call with automatic retry on failure.
    Exponential backoff: 2s, 4s, 8s
    
    Args:
        messages: Chat messages for the model
        model: Model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
    
    Returns:
        Model's response text
    
    Raises:
        Exception after 3 failed attempts
    """
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages
    )
    return response.choices[0].message.content.strip()


def validate_chunks(chunks: List[Dict]) -> tuple:
    """
    Validate chunk structure and filter out malformed chunks.
    
    Returns:
        (valid_chunks, invalid_count, issues)
    """
    valid = []
    invalid_count = 0
    issues = []
    
    for i, chunk in enumerate(chunks):
        errors = []
        
        # Check required fields
        if "text" not in chunk or not chunk["text"].strip():
            errors.append("missing_or_empty_text")
        
        if "metadata" not in chunk:
            errors.append("missing_metadata")
        
        if not isinstance(chunk.get("metadata", {}), dict):
            errors.append("malformed_metadata")
        
        # Check metadata fields
        metadata = chunk.get("metadata", {})
        if not metadata.get("title"):
            errors.append("missing_title")
        
        if errors:
            invalid_count += 1
            issues.append({
                "chunk_index": i,
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "errors": errors
            })
        else:
            valid.append(chunk)
    
    return valid, invalid_count, issues


def format_agent_output(
    role: str,
    content: str,
    chunks: List[Dict] = None,
    error: bool = False
) -> Dict:
    """
    Standardized output format for all agents.
    Ensures consistent structure for downstream processing.
    """
    return {
        "role": role,
        "content": content,
        "chunks": chunks or [],
        "error": error,
        "chunk_count": len(chunks) if chunks else 0
    }


def extract_citations_from_argument(argument: str, chunks: List[Dict]) -> list:
    """
    Parse argument text and map claims to supporting chunks.
    Helpful for traceability and debugging.
    """
    import re
    
    citations = []
    
    # Look for [N] or [Source: N] patterns
    pattern = r'\[(?:Source:\s*)?(\d+)\]'
    
    for match in re.finditer(pattern, argument):
        chunk_num = int(match.group(1))
        if 0 < chunk_num <= len(chunks):
            chunk = chunks[chunk_num - 1]
            citations.append({
                "chunk_num": chunk_num,
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("metadata", {}).get("title"),
                "span": match.span(),
                "relevance": chunk.get("relevance_score", 0)
            })
    
    return citations


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count (1 token ≈ 4 characters).
    For pre-flight checks before API calls.
    """
    return len(text) // 4


if __name__ == "__main__":
    # Test utilities
    test_chunks = [
        {
            "chunk_id": "test-1",
            "text": "Sample evidence text",
            "metadata": {"title": "Test Paper", "authors": "Smith et al.", "url": "http://test.com", "source": "arxiv"}
        }
    ]
    
    print("Testing build_context:")
    print(build_context(test_chunks))
    
    print("\nTesting validate_chunks:")
    valid, invalid, issues = validate_chunks(test_chunks)
    print(f"Valid: {len(valid)}, Invalid: {invalid}")
    
    print("\nTesting estimate_tokens:")
    print(f"'Hello world' → {estimate_tokens('Hello world')} tokens (actual ~2)")