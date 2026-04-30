"""Узел 2 LangGraph: Оценка резюме через LLM с RAG (Retrieval-Augmented Generation)"""

import re
import json
import time
from src.state import ResumeState
from src.config import get_openai_client, MODEL_NAME, TEMPERATURE, MAX_TOKENS
from src.rag_memory import retrieve_similar_examples

client = get_openai_client()

# Базовый промт
BASE_SCORING_PROMPT = """
Ты — строгий HR-эксперт. Твоя задача — объективно оценить резюме. Твоя задача — вернуть ТОЛЬКО JSON. НЕ используй <think> теги. НЕ добавляй пояснения. НЕ пиши ничего до или после JSON.

РЕЗЮМЕ КАНДИДАТА:
{resume_text}

=== ПРАВИЛА ОЦЕНКИ ===

**ВАЖНО: Если информация отсутствует — ставь 0 баллов. Не домысливай и не ставь средние оценки.**

Шкала для каждого критерия:

1. HARD SKILLS (0-35):
   - 0-7: Нет релевантных технологий или только "Word, Excel"
   - 8-14: Базовые знания (1-2 технологии упомянуты)
   - 15-21: Хороший стек (3-4 технологии, современные)
   - 22-28: Сильный стек (5+ технологий, есть глубина)
   - 29-35: Экспертный уровень (редкие технологии, сертификаты)

2. SOFT SKILLS (0-25):
   - 0-5: Нет упоминаний о коммуникации или лидерстве
   - 6-10: Есть общие фразы ("коммуникабельный", "ответственный")
   - 11-17: Есть конкретные примеры ("работа в команде", "помощь коллегам")
   - 18-25: Есть явные маркеры лидерства ("руководил", "менторил", "координировал")

3. ОПЫТ (0-25):
   - 0-5: Нет релевантного опыта или только нерелевантный
   - 6-12: Есть опыт, но нет конкретных результатов
   - 13-18: Есть опыт и цифры/достижения
   - 19-25: Богатый опыт, карьерный рост, измеримые результаты

4. АДАПТИВНОСТЬ (0-15):
   - 0-3: Нет курсов, сертификатов, обучения
   - 4-7: Есть упоминания о курсах (без дат или старые)
   - 8-11: Регулярное обучение, сертификаты за последние 2 года
   - 12-15: Постоянное саморазвитие, смена стеков, пет-проекты
   
Пусть финальный результат не будет 62/72/82/92 - думай больше о каждом данном балле лучше нечетные числа.
Используй точные числа, не округляй до 5 или 10.

=== ПРИМЕРЫ ОЦЕНОК ===

Пример 1 (СЛАБЫЙ кандидат):
Резюме: "Ищу работу, знаю Excel, ответственный, быстро обучаюсь"
Ожидаемая оценка: hard=5, soft=6, experience=3, adaptability=4

Пример 2 (СРЕДНИЙ кандидат):
Резюме: "Python 2 года, Django, SQL, работал в команде, участвовал в проектах"
Ожидаемая оценка: hard=18, soft=14, experience=12, adaptability=8

Пример 3 (СИЛЬНЫЙ кандидат):
Резюме: "Python 5 лет, Django, PostgreSQL, руководил командой, увеличил производительность на 40%, сертификат AWS"
Ожидаемая оценка: hard=28, soft=20, experience=22, adaptability=12

=== ФОРМАТ ОТВЕТА ===

Верни ТОЛЬКО JSON:

{{"hard_skills": число, "soft_skills": число, "experience": число, "adaptability": число, "total_score": число, "recommendation": "Strong Hire/Hire/Consider/Pass", "explanation": "Краткое обоснование"}}

ВАЖНО: total_score = hard_skills + soft_skills + experience + adaptability
"""


def build_prompt_with_rag(resume_text: str) -> str:
    """Формирует промт с RAG примерами"""
    similar_examples = retrieve_similar_examples(resume_text, n_results=2)

    rag_section = ""
    if similar_examples:
        rag_section = "\n\n### ПРИМЕРЫ УСПЕШНЫХ ОЦЕНОК:\n\n"
        for i, example in enumerate(similar_examples, 1):
            rag_section += f"Пример {i}:\n{example}\n\n"
        rag_section += "---\n"

    return rag_section + BASE_SCORING_PROMPT.format(resume_text=resume_text[:2500])


def call_llm_with_retry(prompt: str, resume_text: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            print(f"🔄 Попытка {attempt + 1} вызова Groq API...")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            raw_text = response.choices[0].message.content
            print(f"📝 Сырой ответ: {raw_text[:200]}...")

            raw_text = re.sub(r'```json\s*', '', raw_text)
            raw_text = re.sub(r'```\s*', '', raw_text)
            raw_text = raw_text.strip()

            start = raw_text.find('{')
            end = raw_text.rfind('}')

            if start == -1 or end == -1:
                raise ValueError("JSON не найден")

            json_str = raw_text[start:end + 1]
            result = json.loads(json_str)

            # Получаем баллы от LLM
            hard = result.get("hard_skills", 15)
            soft = result.get("soft_skills", 10)
            exp = result.get("experience", 10)
            adapt = result.get("adaptability", 5)

            # Получаем total_score от LLM или вычисляем
            total = result.get("total_score", hard + soft + exp + adapt)
            total = min(100, max(0, total))

            print(f"✅ LLM вернул: hard={hard}, soft={soft}, exp={exp}, adapt={adapt}, total={total}")

            return {
                "total_score": total,
                "recommendation": result.get("recommendation", "Consider"),
                "scores": {
                    "hard_skills": min(35, max(0, hard)),
                    "soft_skills": min(25, max(0, soft)),
                    "experience": min(25, max(0, exp)),
                    "adaptability": min(15, max(0, adapt))
                },
                "explanation": result.get("explanation", "Анализ выполнен")[:300]
            }

        except Exception as e:
            print(f"❌ LLM Error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

    print("⚠️ Используем эвристический fallback")
    return heuristic_scoring_fallback(resume_text)


def heuristic_scoring_fallback(resume_text: str) -> dict:
    """Fallback при ошибке API"""
    text_lower = resume_text.lower()

    hard_score = 15
    if any(kw in text_lower for kw in ["python", "java", "javascript", "go", "c++"]):
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

    return {
        "total_score": total,
        "recommendation": rec,
        "scores": {
            "hard_skills": hard_score,
            "soft_skills": soft_score,
            "experience": exp_score,
            "adaptability": adapt_score
        },
        "explanation": "⚠️ Оценка на основе ключевых слов (API временно недоступен)"
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

    print(f"\n🔍 Анализ резюме: {state['file_name']}")
    prompt = build_prompt_with_rag(state["cleaned_text"])
    result = call_llm_with_retry(prompt, state["cleaned_text"])

    state["scores"] = result["scores"]
    state["total_score"] = result["total_score"]
    state["recommendation"] = result["recommendation"]
    state["explanation"] = result["explanation"]
    state["error"] = ""

    return state
