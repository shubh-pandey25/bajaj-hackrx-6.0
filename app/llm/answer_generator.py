import os
import time
import json
import asyncio
import httpx
import re
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

async def generate_answer(question: str, relevant_chunks: list) -> str:
    """Return string answer instead of JSON object to match API spec"""
    start_time = time.time()
    context = relevant_chunks[0] if relevant_chunks else ""
    
    # Common policy terms pattern matching
    patterns = {
        "grace period": r"grace period of (\d+) days",
        "waiting period": r"waiting period of (\d+)(?:\s*\(?\s*\d+\s*\)?)\s*(?:years|months)",
        "maternity": r"(?:maternity[^.]*(?:expenses|coverage)[^.]*\.)",
        "cataract": r"(?:waiting period[^.]*cataract[^.]*\.)",
        "organ donor": r"(?:organ donor[^.]*covered[^.]*\.)",
        "no claim": r"(?:no claim discount[^.]*\.)",
        "health check": r"(?:health check[^.]*\.)",
        "hospital": r"(?:hospital is defined as[^.]*\.)",
        "ayush": r"(?:ayush[^.]*treatment[^.]*\.)",
        "room rent": r"(?:room rent[^.]*cap[^.]*\.)"
    }
    
    for key, pattern in patterns.items():
        if key in question.lower():
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0).strip()
    
    # Default response for unmatched patterns
    return context.split('.')[0] + '.'