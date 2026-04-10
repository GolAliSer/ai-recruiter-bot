"""Узел 3 LangGraph: Предложение подходящей роли через LLM"""

import json
import re
import time
from src.state import ResumeState
from src.config import get_openai_client, MODEL_NAME, TEMPERATURE

client = get_openai_client()

ROLE_ADVISOR_PROMPT = """
Ты — эксперт по IT-рекрутингу. На основе анализа резюме предложи кандидату наиболее подходящую профессиональную роль.

РЕЗЮМЕ КАНДИДАТА:
{resume_text}

РЕЗУЛЬТАТЫ ОЦЕНКИ:
- Hard Skills: {hard_skills}/35
- Soft Skills: {soft_skills}/25
- Опыт: {experience}/25
- Адаптивность: {adaptability}/15
- Общий балл: {total_score}/100
- Рекомендация: {recommendation}

Доступные роли:
- Backend-разработчик
- Frontend-разработчик
- Data Scientist
- DevOps инженер
- QA инженер
- Mobile разработчик
- Аналитик
- Системный администратор
- IT-специалист

Верни ТОЛЬКО JSON:
{{"role": "название_роли", "confidence": число_0_100, "reasoning": "почему эта роль подходит"}}
"""


def call_llm_for_role_advice(resume_text: str, scores: dict, total_score: int, recommendation: str,
                             max_retries: int = 2) -> dict:
    """Вызывает LLM для предложения роли"""

    for attempt in range(max_retries):
        try:
            prompt = ROLE_ADVISOR_PROMPT.format(
                resume_text=resume_text[:2000],
                hard_skills=scores["hard_skills"],
                soft_skills=scores["soft_skills"],
                experience=scores["experience"],
                adaptability=scores["adaptability"],
                total_score=total_score,
                recommendation=recommendation
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250
            )

            raw_text = response.choices[0].message.content
            raw_text = re.sub(r'```json\s*', '', raw_text)
            raw_text = re.sub(r'```\s*', '', raw_text)
            raw_text = raw_text.strip()

            start = raw_text.find('{')
            end = raw_text.rfind('}')

            if start != -1 and end != -1 and end > start:
                json_str = raw_text[start:end + 1]
                result = json.loads(json_str)

                return {
                    "role": result.get("role", "IT-специалист"),
                    "confidence": min(100, max(0, result.get("confidence", 50))),
                    "reasoning": result.get("reasoning", "Анализ выполнен")[:200]
                }

        except Exception as e:
            print(f"LLM Role Advisor error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

    # Fallback
    return {
        "role": "IT-специалист",
        "confidence": 50,
        "reasoning": "Автоматическое определение роли"
    }


def suggest_role(state: ResumeState) -> ResumeState:
    """Предлагает подходящую роль через LLM"""

    if len(state["cleaned_text"]) < 100:
        state["detected_role"] = "IT-специалист"
        state["role_confidence"] = 30
        state["role_reasoning"] = "Текст слишком короткий"
        return state

    result = call_llm_for_role_advice(
        resume_text=state["cleaned_text"],
        scores=state["scores"],
        total_score=state["total_score"],
        recommendation=state["recommendation"]
    )

    state["detected_role"] = result["role"]
    state["role_confidence"] = result["confidence"]
    state["role_reasoning"] = result["reasoning"]

    return state