"""Обработка чата: управление диалогом и действиями"""

import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime
from src.graph import app_graph
from src.utils import extract_text_from_file
from src.database import get_feedback_stats, get_user_history, save_evaluation, save_feedback
from src.rag_memory import add_successful_example


def format_detailed_result(result: dict) -> str:
    """Форматирует детальный результат оценки (без фидбека)"""
    s = result["scores"]

    rec_emoji = {
        "Strong Hire": "🟢",
        "Hire": "🟡",
        "Consider": "🟠",
        "Pass": "🔴"
    }.get(result["recommendation"], "⚪")

    def progress_bar(value, max_val, width=10):
        filled = int(width * value / max_val) if max_val > 0 else 0
        return "█" * filled + "░" * (width - filled)

    output = f"""
**Общий балл:** **{result['total_score']}/100**
**Рекомендация:** {rec_emoji} {result['recommendation']}
**Предполагаемая роль:** {result['detected_role']}

**Детализация:**
| Критерий | Оценка | Визуализация |
|----------|--------|--------------|
| Hard Skills | {s['hard_skills']}/35 | {progress_bar(s['hard_skills'], 35)} |
| Soft Skills | {s['soft_skills']}/25 | {progress_bar(s['soft_skills'], 25)} |
| Опыт | {s['experience']}/25 | {progress_bar(s['experience'], 25)} |
| Адаптивность | {s['adaptability']}/15 | {progress_bar(s['adaptability'], 15)} |

**💬 {result['explanation']}**
"""
    return output


def render_feedback_form(evaluation_id: int, resume_text: str = None, evaluation_result: dict = None):
    """
    Отрисовывает форму фидбека с помощью Streamlit.

    Args:
        evaluation_id: ID оценки в БД
        resume_text: Текст резюме (нужен для RAG)
        evaluation_result: Результат оценки (нужен для RAG)
    """
    if evaluation_id:
        with st.container():
            st.markdown("**📝 Оцените точность этого анализа:**")

            with st.form(key=f"feedback_form_{evaluation_id}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    rating = st.select_slider(
                        "Точность",
                        options=[1, 2, 3, 4, 5],
                        value=5,
                        format_func=lambda x: "⭐" * x + "☆" * (5 - x),
                        label_visibility="collapsed"
                    )
                with col2:
                    comment = st.text_input(
                        "Комментарий (необязательно)",
                        placeholder="Например: Оценка завышена",
                        label_visibility="collapsed"
                    )

                submitted = st.form_submit_button("📤 Отправить фидбек", use_container_width=True)

                if submitted:
                    # 1. Сохраняем фидбек в SQLite
                    save_feedback(evaluation_id, rating, comment)

                    # 2. Если рейтинг высокий (4 или 5) и есть данные для RAG
                    if rating >= 4 and resume_text and evaluation_result:
                        success = add_successful_example(
                            resume_text=resume_text,
                            evaluation_result=evaluation_result,
                            rating=rating
                        )
                        if success:
                            st.success("✅ Спасибо за фидбек! Этот пример сохранён для улучшения будущих оценок.")
                        else:
                            st.success("✅ Спасибо за фидбек!")
                    else:
                        st.success("✅ Спасибо за фидбек!")

                    st.rerun()


def format_results_table(results: list) -> tuple:
    """
    Форматирует таблицу результатов с краткими пояснениями.
    Возвращает (таблица_в_markdown, отсортированный_список_результатов)
    """
    # Сортируем по баллам
    results_sorted = sorted(results, key=lambda x: x["total_score"], reverse=True)

    # Создаем строку таблицы
    table = "| # | Файл | Роль | Балл | Рекомендация | Краткое обоснование |\n"
    table += "|---|------|------|------|--------------|---------------------|\n"

    for i, r in enumerate(results_sorted, 1):
        short_exp = r["explanation"][:60] + "..." if len(r["explanation"]) > 60 else r["explanation"]
        table += f"| {i} | {r['file_name'][:30]} | {r['detected_role'][:15]} | **{r['total_score']}** | {r['recommendation']} | {short_exp} |\n"

    return table, results_sorted


def export_to_csv(results: list) -> bytes:
    """Экспортирует результаты в CSV"""
    df = pd.DataFrame([{
        "Файл": r["file_name"],
        "Роль": r["detected_role"],
        "Балл": r["total_score"],
        "Рекомендация": r["recommendation"],
        "Hard Skills": r["scores"]["hard_skills"],
        "Soft Skills": r["scores"]["soft_skills"],
        "Опыт": r["scores"]["experience"],
        "Адаптивность": r["scores"]["adaptability"],
        "Объяснение": r["explanation"]
    } for r in results])

    # Сортируем по баллам
    df = df.sort_values("Балл", ascending=False)

    return df.to_csv(index=False).encode('utf-8-sig')


def save_evaluation_to_db(user_id: int, result: dict) -> int:
    """Сохраняет оценку в БД и возвращает evaluation_id"""
    return save_evaluation(
        user_id=user_id,
        file_name=result["file_name"],
        detected_role=result["detected_role"],
        role_confidence=result.get("role_confidence", 50),
        role_reasoning=result.get("role_reasoning", ""),
        total_score=result["total_score"],
        recommendation=result["recommendation"],
        scores=result["scores"],
        explanation=result["explanation"],
        word_count=result.get("word_count", 0)
    )


def handle_scoring(user_id: int, file_names: list, uploaded_files: list) -> tuple:
    """
    Обработка запроса на оценку резюме (одиночная или массовая).
    Возвращает (тип_результата, вывод, результаты, evaluation_ids)
    тип_результата: "single", "batch", "error"
    """
    if not uploaded_files:
        return "error", "❌ Пожалуйста, загрузите файл резюме для оценки.", [], []

    results = []
    evaluation_ids = []

    for file in uploaded_files:
        text = extract_text_from_file(file)
        if "Ошибка" not in text:
            result = app_graph.invoke({
                "file_name": file.name,
                "raw_text": text,
                "cleaned_text": "",
                "word_count": 0,
                "detected_role": "",
                "role_confidence": 0,
                "role_reasoning": "",
                "scores": {},
                "total_score": 0,
                "recommendation": "",
                "explanation": "",
                "final_output": "",
                "error": ""
            })
            results.append(result)

            if user_id:
                eval_id = save_evaluation_to_db(user_id, result)
                evaluation_ids.append(eval_id)
            else:
                evaluation_ids.append(None)

    if not results:
        return "error", "❌ Не удалось обработать файлы.", [], []

    # Если один файл — детальный вывод
    if len(results) == 1:
        output = format_detailed_result(results[0])
        return "single", output, results, evaluation_ids

    # Если несколько файлов — таблица
    table, results_sorted = format_results_table(results)
    output = f"### 📊 Результаты оценки ({len(results)} файлов)\n\n{table}\n\n"
    return "batch", output, results_sorted, evaluation_ids


def handle_compare(user_id: int, file_names: list, uploaded_files: list) -> tuple:
    """
    Обработка запроса на сравнение двух резюме.
    Возвращает (вывод, результаты, evaluation_ids)
    """
    if len(uploaded_files) < 2:
        return "❌ Для сравнения нужно загрузить минимум 2 резюме.", [], []

    results = []
    evaluation_ids = []

    for file in uploaded_files[:2]:
        text = extract_text_from_file(file)
        if "Ошибка" not in text:
            result = app_graph.invoke({
                "file_name": file.name,
                "raw_text": text,
                "cleaned_text": "",
                "word_count": 0,
                "detected_role": "",
                "role_confidence": 0,
                "role_reasoning": "",
                "scores": {},
                "total_score": 0,
                "recommendation": "",
                "explanation": "",
                "final_output": "",
                "error": ""
            })
            results.append(result)

            if user_id:
                eval_id = save_evaluation_to_db(user_id, result)
                evaluation_ids.append(eval_id)
            else:
                evaluation_ids.append(None)

    if len(results) < 2:
        return "❌ Не удалось обработать оба файла.", [], []

    winner = results[0] if results[0]["total_score"] >= results[1]["total_score"] else results[1]

    # Таблица сравнения
    comparison = f"""
### ⚖️ Сравнение кандидатов

| Критерий | {results[0]['file_name']} | {results[1]['file_name']} |
|----------|---------------------------|---------------------------|
| **Общий балл** | **{results[0]['total_score']}/100** | **{results[1]['total_score']}/100** |
| Рекомендация | {results[0]['recommendation']} | {results[1]['recommendation']} |
| Предполагаемая роль | {results[0]['detected_role']} | {results[1]['detected_role']} |
| Hard Skills | {results[0]['scores']['hard_skills']}/35 | {results[1]['scores']['hard_skills']}/35 |
| Soft Skills | {results[0]['scores']['soft_skills']}/25 | {results[1]['scores']['soft_skills']}/25 |
| Опыт | {results[0]['scores']['experience']}/25 | {results[1]['scores']['experience']}/25 |
| Адаптивность | {results[0]['scores']['adaptability']}/15 | {results[1]['scores']['adaptability']}/15 |

🏆 **Победитель:** {winner['file_name']} ({winner['total_score']}/100)

---

### 📝 Обоснование:

**{results[0]['file_name']}:** {results[0]['explanation']}

**{results[1]['file_name']}:** {results[1]['explanation']}
"""
    return comparison, results, evaluation_ids


def handle_stats(user_id: int = None) -> str:
    """Обработка запроса на статистику"""
    stats = get_feedback_stats()

    response = f"""
## 📊 Статистика системы

| Показатель | Значение |
|------------|----------|
| Всего фидбека | {stats['total']} |
| Средняя оценка | {stats['avg_rating']}/5 |
"""

    if user_id:
        history = get_user_history(user_id, limit=5)
        if history:
            response += "\n### 📜 Ваши последние оценки:\n\n"
            for h in history:
                response += f"- **{h['file'][:30]}...** → {h['score']}/100 ({h['rec']})\n"

    return response


def handle_help() -> str:
    """Справка по командам и возможностям"""
    return """
## 🤖 Что я умею?

### 📄 Оценка резюме
- `Оцени` — оценить загруженные файлы
- `Оцени resume.pdf` — оценить конкретный файл

### ⚖️ Сравнение кандидатов
- `Сравни` — сравнить загруженные файлы
- `Кто лучше?` — сравнить двух кандидатов

### 📊 Статистика
- `Покажи статистику` — общая статистика системы
- `Моя история` — ваши последние оценки

### 💬 Команды
- `/help` — эта справка
- `/clear` — очистить чат
- `/stats` — статистика системы
"""


def handle_chat(question: str) -> str:
    """Обычный чат (приветствия, общие вопросы)"""
    q_lower = question.lower()

    if any(kw in q_lower for kw in ["привет", "здравствуй", "hi", "hello"]):
        return "👋 Привет! Я AI-рекрутер. Загрузите резюме, и я помогу с оценкой. Напишите **/help** для справки."
    elif any(kw in q_lower for kw in ["как дела", "how are you"]):
        return "✅ Всё отлично! Готов анализировать резюме. Загружайте файлы!"
    elif any(kw in q_lower for kw in ["спасиб", "thanks"]):
        return "🙏 Пожалуйста! Обращайтесь ещё."
    elif any(kw in q_lower for kw in ["что ты умеешь", "функции", "возможности"]):
        return handle_help()
    else:
        return f"Я не совсем понял. Загрузите резюме для оценки, или напишите **/help** для справки."
