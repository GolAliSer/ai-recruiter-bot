"""Работа с SQLite базой данных: пользователи, оценки, фидбек"""

import sqlite3
import json
import os
import bcrypt
from datetime import datetime
from src.config import DB_PATH


def init_database():
    """Инициализирует базу данных (создает таблицы если их нет)"""
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Таблица пользователей (с паролем)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            is_active INTEGER DEFAULT 1,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Таблица оценок резюме
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

    # Таблица для RAG примеров (подготовка для фазы 2)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_text TEXT,
            scores_json TEXT,
            total_score INTEGER DEFAULT 0,
            recommendation TEXT DEFAULT '',
            rating INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ База данных инициализирована")


def hash_password(password: str) -> str:
    """Хеширует пароль"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Проверяет пароль"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


def register_user(username: str, password: str) -> tuple:
    """
    Регистрирует нового пользователя.
    Возвращает (success, user_id, message)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Проверяем, существует ли пользователь
    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    if cursor.fetchone():
        conn.close()
        return False, None, "Пользователь с таким именем уже существует"

    # Создаём нового пользователя
    password_hash = hash_password(password)
    cursor.execute('''
        INSERT INTO users (username, password_hash)
        VALUES (?, ?)
    ''', (username, password_hash))

    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return True, user_id, "Регистрация успешна"


def login_user(username: str, password: str) -> tuple:
    """
    Вход пользователя.
    Возвращает (success, user_id, message)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT id, password_hash, is_active FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        return False, None, "Неверное имя пользователя или пароль"

    user_id, password_hash, is_active = result

    if not is_active:
        conn.close()
        return False, None, "Учётная запись заблокирована"

    if not verify_password(password, password_hash):
        conn.close()
        return False, None, "Неверное имя пользователя или пароль"

    # Обновляем время последнего входа
    cursor.execute('UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()

    return True, user_id, f"Добро пожаловать, {username}!"


def get_or_create_user(username: str) -> int:
    """
    [DEPRECATED] Старая функция для простой авторизации.
    Оставлена для совместимости, но не используется.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()

    if result:
        user_id = result[0]
        cursor.execute('UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
    else:
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, ""))
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
