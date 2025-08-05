# app/llm/answer_generator.py

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

async def generate_answer(question: str, relevant_chunks: list[str], answer_only: bool = False):
    """
    Performs retrieval-augmented generation over the provided chunks.
    If answer_only is True, returns just the answer string. Otherwise returns dict.
    """
    # 1. Build context—join up to 5 chunks
    top_chunks = relevant_chunks[:5]
    context = "\n\n---\n\n".join(top_chunks)

    # 2. Craft the prompt
    prompt = f"""
You are a health-insurance policy expert.  Use ONLY the information in the “CONTEXT” below.
If the policy does not cover the question, say “Not covered under this policy.”

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely with reference to the relevant section if possible.
"""

    # 3. Call the LLM
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":       "gpt-4o-mini",     # or your preferred model
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens":  300,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    answer_text = data["choices"][0]["message"]["content"].strip()

    if answer_only:
        return answer_text
    return {
        "question": question,
        "answer":   answer_text
    }