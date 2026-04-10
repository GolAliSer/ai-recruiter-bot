"""Узел 2 LangGraph: Оценка резюме через LLM"""

import re
import json
import time
from src.state import ResumeState
from src.config import get_openai_client, MODEL_NAME, TEMPERATURE, MAX_TOKENS

client = get_openai_client()

SCORING_PROMPT = """
Ты — эксперт по оценке IT-резюме. Оцени кандидата по 4 критериям.

РЕЗЮМЕ КАНДИДАТА:
{resume_text}

=== КРИТЕРИИ ОЦЕНКИ ===

1. HARD SKILLS (0-35) — технологии, языки, инструменты, сертификаты
2. SOFT SKILLS (0-25) — коммуникация, лидерство, работа в команде
3. ОПЫТ (0-25) — релевантность, достижения, карьерный рост
4. АДАПТИВНОСТЬ (0-15) — обучение, новые технологии, курсы

=== ТРЕБОВАНИЯ К ОТВЕТУ ===

Верни ТОЛЬКО JSON. В поле "explanation" напиши ЧЕЛОВЕКО-ЧИТАЕМОЕ ОБЪЯСНЕНИЕ (2-3 предложения) на русском языке.

Формат ответа:
{{"total_score": число, "recommendation": "Strong Hire/Hire/Consider/Pass", "scores": {{"hard_skills": число, "soft_skills": число, "experience": число, "adaptability": число}}, "explanation": "Человеко-читаемое объяснение на русском"}}
"""


def call_llm_with_retry(prompt: str, resume_text: str, max_retries: int = 3) -> dict:
    """Вызов LLM с повторными попытками"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            raw_text = response.choices[0].message.content

            # Очищаем ответ
            raw_text = re.sub(r'```json\s*', '', raw_text)
            raw_text = re.sub(r'```\s*', '', raw_text)
            raw_text = raw_text.strip()

            # Ищем JSON
            start = raw_text.find('{')
            end = raw_text.rfind('}')

            if start != -1 and end != -1 and end > start:
                json_str = raw_text[start:end + 1]
                result = json.loads(json_str)
            else:
                raise ValueError("JSON not found")

            scores = result.get("scores", {})
            calculated_total = sum(scores.values())

            return {
                "total_score": min(100, max(0, calculated_total)),
                "recommendation": result.get("recommendation", "Consider"),
                "scores": {
                    "hard_skills": min(35, max(0, scores.get("hard_skills", 15))),
                    "soft_skills": min(25, max(0, scores.get("soft_skills", 10))),
                    "experience": min(25, max(0, scores.get("experience", 10))),
                    "adaptability": min(15, max(0, scores.get("adaptability", 5)))
                },
                "explanation": result.get("explanation", "Анализ выполнен")[:300]
            }
        except Exception as e:
            error_msg = str(e)
            print(f"LLM Error (attempt {attempt + 1}): {error_msg}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

    # Fallback при ошибке API
    return heuristic_scoring_fallback(resume_text)


def heuristic_scoring_fallback(resume_text: str) -> dict:
    """Fallback при недоступности API"""
    text_lower = resume_text.lower()

    # Эвристическая оценка
    hard_score = 15
    if any(kw in text_lower for kw in ["python", "java", "javascript", "go", "c++", "kotlin", "swift"]):
        hard_score += 10
    if any(kw in text_lower for kw in ["sql", "docker", "git", "linux"]):
        hard_score += 5
    hard_score = min(35, hard_score)

    soft_score = 12
    if any(kw in text_lower for kw in ["коммуникац", "лидер", "команд", "team"]):
        soft_score += 8
    soft_score = min(25, soft_score)

    exp_score = 10
    years_match = re.search(r'(\d+)\s*(?:год|лет|года|year)', text_lower)
    if years_match:
        years = int(years_match.group(1))
        if years >= 5:
            exp_score = 20
        elif years >= 3:
            exp_score = 15

    adapt_score = 8
    if any(kw in text_lower for kw in ["курс", "обучени", "сертификат"]):
        adapt_score += 5
    adapt_score = min(15, adapt_score)

    total = hard_score + soft_score + exp_score + adapt_score

    if total >= 85:
        rec = "Strong Hire"
    elif total >= 70:
        rec = "Hire"
    elif total >= 50:
        rec = "Consider"
    else:
        rec = "Pass"

    explanation = f"⚠️ API временно недоступен. Оценка выполнена на основе ключевых слов."

    return {
        "total_score": total,
        "recommendation": rec,
        "scores": {
            "hard_skills": hard_score,
            "soft_skills": soft_score,
            "experience": exp_score,
            "adaptability": adapt_score
        },
        "explanation": explanation
    }


def scorer_node(state: ResumeState) -> ResumeState:
    """LLM оценка резюме"""

    if len(state["cleaned_text"]) < 100:
        state["scores"] = {"hard_skills": 10, "soft_skills": 8, "experience": 5, "adaptability": 3}
        state["total_score"] = 26
        state["recommendation"] = "Pass"
        state["explanation"] = "Резюме слишком короткое"
        state["error"] = ""
        return state

    prompt = SCORING_PROMPT.format(
        resume_text=state["cleaned_text"][:2500]
    )

    result = call_llm_with_retry(prompt, state["cleaned_text"])

    state["scores"] = result["scores"]
    state["total_score"] = result["total_score"]
    state["recommendation"] = result["recommendation"]
    state["explanation"] = result["explanation"]
    state["error"] = ""

    return state
