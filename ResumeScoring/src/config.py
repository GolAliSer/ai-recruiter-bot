"""Конфигурация приложения: API ключи, константы, настройки"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем .env
load_dotenv()

# API ключи
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Выбор провайдера: "groq" или "gemini"
PROVIDER = "groq"  # меняйте здесь

# Настройки в зависимости от провайдера
if PROVIDER == "groq":
    # Groq - быстрый, 1000 запросов/день бесплатно
    API_KEY = GROQ_API_KEY
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL_NAME = "llama-3.3-70b-versatile"  # или "llama-3.1-8b-instant" (быстрее)
    TEMPERATURE = 0.3
    MAX_TOKENS = 800
else:
    # Gemini (запасной вариант)
    API_KEY = GEMINI_API_KEY
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    MODEL_NAME = "gemini-2.5-flash-lite"
    TEMPERATURE = 0.3
    MAX_TOKENS = 800

# Проверка ключа
if not API_KEY:
    raise ValueError(f"❌ API ключ для {PROVIDER.upper()} не найден в файле .env")

# Настройки базы данных
DB_PATH = "data/feedback.db"

# Клиент OpenAI
def get_openai_client():
    """Возвращает настроенный клиент OpenAI"""
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )