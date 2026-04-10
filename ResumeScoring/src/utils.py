"""Вспомогательные функции: парсинг файлов, очистка текста"""

import re
import pdfplumber
from docx import Document


def extract_text_from_file(uploaded_file) -> str:
    """Извлекает текст из загруженного файла (PDF, DOCX, TXT)"""
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == 'pdf':
            with pdfplumber.open(uploaded_file) as pdf:
                text = '\n'.join([p.extract_text() or '' for p in pdf.pages])
        elif ext == 'docx':
            doc = Document(uploaded_file)
            text = '\n'.join([p.text for p in doc.paragraphs])
        else:
            text = uploaded_file.getvalue().decode('utf-8')

        if len(text.strip()) < 50:
            return "Ошибка: текст слишком короткий"
        return text
    except Exception as e:
        return f"Ошибка: {str(e)}"


def clean_text(text: str) -> str:
    """Очищает текст от мусора"""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s\.\,\-\!\?\(\)\:\;\"\'\%\$\€\#\+]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_years_experience(text: str) -> int:
    """Извлекает годы опыта из текста"""
    text_lower = text.lower()
    years_match = re.search(r'(\d+)\s*(?:год|лет|года|year)', text_lower)
    if years_match:
        return int(years_match.group(1))
    return 0