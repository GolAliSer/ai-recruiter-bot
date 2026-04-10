"""Определение намерений пользователя из текста"""

import json
import re
from src.config import get_openai_client, MODEL_NAME

client = get_openai_client()

INTENT_PROMPT = """
Ты — AI-ассистент рекрутера. Определи, что хочет сделать пользователь.

Сообщение пользователя: {user_message}

Возможные намерения:
- "scoring" — пользователь хочет ОЦЕНИТЬ резюме (упоминает файл, резюме, кандидата)
- "compare" — пользователь хочет СРАВНИТЬ два или более резюме
- "stats" — пользователь хочет увидеть СТАТИСТИКУ (оценки, фидбек)
- "help" — пользователь просит ПОМОЩЬ (что ты умеешь, команды)
- "chat" — обычный РАЗГОВОР (приветствие, вопрос о возможностях)

Также извлеки из сообщения:
- file_names: список имен файлов, если упоминаются (например, "resume1.pdf", "резюме Ивана")
- question: конкретный вопрос, если есть

Верни ТОЛЬКО JSON:
{{"intent": "scoring/compare/stats/help/chat", "file_names": ["файл1.pdf", "файл2.pdf"], "question": "уточняющий вопрос или пустая строка"}}
"""


def classify_intent(user_message: str) -> dict:
    """Определяет намерение пользователя"""
    try:
        prompt = INTENT_PROMPT.format(user_message=user_message[:500])

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )

        raw_text = response.choices[0].message.content
        raw_text = re.sub(r'```json\s*', '', raw_text)
        raw_text = re.sub(r'```\s*', '', raw_text)

        start = raw_text.find('{')
        end = raw_text.rfind('}')

        if start != -1 and end != -1:
            json_str = raw_text[start:end + 1]
            result = json.loads(json_str)
            return {
                "intent": result.get("intent", "chat"),
                "file_names": result.get("file_names", []),
                "question": result.get("question", "")
            }
    except Exception as e:
        print(f"Intent classification error: {e}")

    # Fallback: определяем по ключевым словам
    msg_lower = user_message.lower()
    if any(kw in msg_lower for kw in ["оцени", "проанализируй", "скоринг", "оценка"]):
        return {"intent": "scoring", "file_names": [], "question": ""}
    elif any(kw in msg_lower for kw in ["сравни", "сравнение", "compare"]):
        return {"intent": "compare", "file_names": [], "question": ""}
    elif any(kw in msg_lower for kw in ["статистик", "stats", "сколько"]):
        return {"intent": "stats", "file_names": [], "question": ""}
    elif any(kw in msg_lower for kw in ["помощь", "help", "умеешь", "команды"]):
        return {"intent": "help", "file_names": [], "question": ""}
    else:
        return {"intent": "chat", "file_names": [], "question": ""}