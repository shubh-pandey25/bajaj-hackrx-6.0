# Common helper functions (if needed) can be placed here

"""
This module is intended for reusable utility functions
across the LLM-powered query-retrieval system.
"""

from typing import Dict, List

def calculate_score(answer: Dict, question_weight: float, doc_weight: float) -> float:
    """Calculate weighted score for an answer"""
    if answer["confidence_score"] > 80:
        return question_weight * doc_weight
    elif answer["confidence_score"] > 50:
        return 0.5 * question_weight * doc_weight
    return 0

def get_document_weight(doc_id: str, known_docs: List[str]) -> float:
    """Get document weight based on if it's known/unknown"""
    return 0.5 if doc_id in known_docs else 2.0

def get_question_weight(question: str) -> float:
    """Get question weight based on complexity"""
    complex_keywords = ["waiting period", "conditions", "define", "coverage", "sub-limits"]
    return 2.0 if any(k in question.lower() for k in complex_keywords) else 1.0
