from groq import Groq
from typing import Dict
from dotenv import load_dotenv
import os
load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an impartial academic judge evaluating a research debate.
Your job is to assess both sides fairly based purely on evidence quality,
argument structure, and citation strength.
Rules:
- Do not favour either side by default
- Base your verdict only on the arguments and evidence presented
- Be specific about which arguments were stronger and why
- Always provide a confidence score between 0.0 and 1.0
"""

def run_judge_agent(
    topic          : str,
    for_argument   : str,
    against_argument: str,
    for_rebuttal   : str,
    against_rebuttal: str
) -> Dict:
    user_message = f"""
Topic: {topic}

--- FOR ARGUMENT ---
{for_argument}

--- AGAINST ARGUMENT ---
{against_argument}

--- FOR REBUTTAL ---
{for_rebuttal}

--- AGAINST REBUTTAL ---
{against_rebuttal}

Based on the full debate above, deliver your verdict.
Structure your response as:
1. Winner        : FOR or AGAINST
2. Confidence    : score between 0.0 and 1.0
3. Reasoning     : why this side won
4. Strongest point from winner
5. Weakest point from loser
6. Overall summary
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    return {
        "role"    : "JUDGE",
        "verdict" : response.choices[0].message.content,
        "topic"   : topic
    }

if __name__ == "__main__":
    print("Judge agent loaded. Run via orchestrator.")
