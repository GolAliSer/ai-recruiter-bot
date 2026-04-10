"""Узел 1 LangGraph: Очистка и подготовка текста резюме"""

from src.utils import clean_text
from src.state import ResumeState


def parser_node(state: ResumeState) -> ResumeState:
    """Очистка и подготовка текста резюме"""
    text = state["raw_text"]
    cleaned = clean_text(text)

    state["cleaned_text"] = cleaned[:3000]
    state["word_count"] = len(cleaned.split())
    state["error"] = ""
    return state