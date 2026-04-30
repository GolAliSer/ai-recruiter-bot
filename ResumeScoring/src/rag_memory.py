"""Упрощённый RAG на основе ключевых слов (без эмбеддингов)"""

import json
import sqlite3
from src.config import DB_PATH


def init_rag_table():
    """Инициализирует таблицу для хранения успешных примеров"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_text TEXT,
            scores_json TEXT,
            total_score INTEGER,
            recommendation TEXT,
            rating INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def add_successful_example(resume_text: str, evaluation_result: dict, rating: int):
    """
    Сохраняет успешный кейс (фидбек 4 или 5) в базу RAG.
    """
    if rating < 4:
        return False

    init_rag_table()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO rag_examples (resume_text, scores_json, total_score, recommendation, rating)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        resume_text[:3000],
        json.dumps(evaluation_result.get("scores", {})),
        evaluation_result.get("total_score", 0),
        evaluation_result.get("recommendation", ""),
        rating
    ))

    conn.commit()
    conn.close()
    print(f"✅ RAG: Успешный пример сохранён (рейтинг {rating})")
    return True


def retrieve_similar_examples(resume_text: str, n_results: int = 2) -> list:
    """
    Ищет похожие примеры по ключевым словам (без эмбеддингов).
    Возвращает список строк с примерами для промта.
    """
    init_rag_table()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Извлекаем ключевые слова из текста (первые 100 слов)
    keywords = resume_text.lower().split()[:100]
    # Убираем стоп-слова
    stop_words = {'и', 'в', 'на', 'с', 'по', 'к', 'у', 'за', 'из', 'от', 'до', 'о', 'об', 'для', 'без', 'через', 'над',
                  'под'}
    keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 3]

    if not keywords:
        print("⚠️ RAG: Недостаточно ключевых слов для поиска")
        return []

    # Простой поиск по ключевым словам (SQL LIKE)
    # Берём последние 10 успешных примеров
    cursor.execute('''
        SELECT resume_text, scores_json, total_score, recommendation, rating
        FROM rag_examples
        ORDER BY created_at DESC
        LIMIT 10
    ''')

    examples = cursor.fetchall()
    conn.close()

    if not examples:
        print("⚠️ RAG: Нет сохранённых примеров")
        return []

    # Оцениваем похожесть по количеству совпадающих ключевых слов
    scored_examples = []
    for ex in examples:
        ex_text = ex[0].lower()
        matches = sum(1 for kw in keywords if kw in ex_text)
        if matches > 0:
            scored_examples.append((matches, ex))

    # Сортируем по количеству совпадений и берём топ-n_results
    scored_examples.sort(key=lambda x: x[0], reverse=True)

    result_examples = []
    for i, (matches, ex) in enumerate(scored_examples[:n_results]):
        scores = json.loads(ex[1])
        example_text = f"""
**Пример {i + 1} (совпадений: {matches}):**
Резюме: {ex[0][:500]}...
Оценка: {ex[2]}/100 ({ex[3]})
Детали: Hard: {scores.get('hard_skills', 0)}/35, Soft: {scores.get('soft_skills', 0)}/25
"""
        result_examples.append(example_text)

    if result_examples:
        print(f"✅ RAG: Найдено {len(result_examples)} похожих примеров")
    else:
        print("⚠️ RAG: Похожих примеров не найдено")

    return result_examples