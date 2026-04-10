"""Работа с SQLite базой данных: пользователи, оценки, фидбек"""

import sqlite3
import json
import os
from datetime import datetime
from src.config import DB_PATH


def init_database():
    """Инициализирует базу данных (создает таблицы если их нет)"""
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Таблица пользователей
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Таблица оценок резюме (с новыми колонками)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            file_name TEXT,
            detected_role TEXT,
            role_confidence INTEGER DEFAULT 0,
            role_reasoning TEXT,
            total_score INTEGER,
            recommendation TEXT,
            scores_json TEXT,
            explanation TEXT,
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Таблица фидбека
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id INTEGER,
            rating INTEGER,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ База данных инициализирована")


def get_or_create_user(username: str) -> int:
    """Получает или создает пользователя, возвращает user_id"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()

    if result:
        user_id = result[0]
        cursor.execute('UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
    else:
        cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
        user_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return user_id


def save_evaluation(user_id: int, file_name: str, detected_role: str,
                    role_confidence: int, role_reasoning: str,
                    total_score: int, recommendation: str, scores: dict,
                    explanation: str, word_count: int) -> int:
    """Сохраняет оценку резюме, возвращает evaluation_id"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO evaluations (user_id, file_name, detected_role, role_confidence, role_reasoning,
                                 total_score, recommendation, scores_json, explanation, word_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, file_name, detected_role, role_confidence, role_reasoning,
          total_score, recommendation, json.dumps(scores), explanation, word_count))

    evaluation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return evaluation_id


def save_feedback(evaluation_id: int, rating: int, comment: str):
    """Сохраняет фидбек"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO feedback (evaluation_id, rating, comment)
        VALUES (?, ?, ?)
    ''', (evaluation_id, rating, comment))

    conn.commit()
    conn.close()


def get_user_history(user_id: int, limit: int = 10) -> list:
    """Возвращает историю оценок пользователя"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, file_name, total_score, recommendation, created_at
        FROM evaluations
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (user_id, limit))

    results = cursor.fetchall()
    conn.close()

    return [{"id": r[0], "file": r[1], "score": r[2], "rec": r[3], "date": r[4]} for r in results]


def get_feedback_stats() -> dict:
    """Возвращает статистику по фидбеку"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM feedback')
    total_feedback = cursor.fetchone()[0]

    cursor.execute('SELECT AVG(rating) FROM feedback')
    avg_rating = cursor.fetchone()[0]

    conn.close()
    return {"total": total_feedback, "avg_rating": round(avg_rating, 1) if avg_rating else 0}