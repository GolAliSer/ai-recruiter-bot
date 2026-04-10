"""Определение состояния LangGraph"""

from typing import TypedDict, Dict, Any


class ResumeState(TypedDict):
    """Состояние графа обработки резюме"""
    file_name: str
    raw_text: str
    cleaned_text: str
    word_count: int
    detected_role: str
    role_confidence: int      # НОВОЕ: уверенность в определении роли (0-100)
    role_reasoning: str       # НОВОЕ: обоснование выбора роли
    scores: Dict[str, int]
    total_score: int
    recommendation: str
    explanation: str
    final_output: str
    error: str