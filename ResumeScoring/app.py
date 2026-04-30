"""Главный файл приложения: Чат-бот с естественным языком"""

import streamlit as st
from datetime import datetime
from src.config import MODEL_NAME
from src.database import (
    init_database, register_user, login_user,
    save_evaluation, save_feedback,
    get_user_history, get_feedback_stats
)
from src.intent_classifier import classify_intent
from src.chat_handler import (
    handle_scoring, handle_compare, handle_stats,
    handle_help, handle_chat, render_feedback_form,
    format_detailed_result, export_to_csv
)
from src.graph import app_graph
from src.state import ResumeState

# Инициализация
init_database()

# ============================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================

st.set_page_config(page_title="AI Recruiter Bot", page_icon="🤖", layout="wide")

st.title("🤖 AI Recruiter Bot")
st.caption(f"Чат-бот с естественным языком | LangGraph | {MODEL_NAME}")

# ============================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ============================================

# Авторизация
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.auth_mode = "login"

# Результаты анализа (унифицированные для всех типов)
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_eval_ids" not in st.session_state:
    st.session_state.current_eval_ids = None

# История чата
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "👋 Привет! Я AI-рекрутер. Загрузите файлы резюме и напишите, что нужно сделать: оценить, сравнить или посмотреть статистику."}
    ]

# Загруженные файлы
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ============================================
# ФУНКЦИИ АВТОРИЗАЦИИ
# ============================================

def show_auth_form():
    """Отображает форму логина/регистрации"""
    with st.sidebar:
        st.markdown("### 👤 Вход / Регистрация")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Вход", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "login" else "secondary"):
                st.session_state.auth_mode = "login"
                st.rerun()
        with col2:
            if st.button("Регистрация", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "register" else "secondary"):
                st.session_state.auth_mode = "register"
                st.rerun()

        st.markdown("---")

        username = st.text_input("Имя пользователя", placeholder="Введите имя")
        password = st.text_input("Пароль", type="password", placeholder="Введите пароль")

        if st.session_state.auth_mode == "register":
            password_confirm = st.text_input("Подтвердите пароль", type="password", placeholder="Повторите пароль")

            if st.button("Зарегистрироваться", use_container_width=True):
                if not username or not password:
                    st.error("Заполните все поля")
                elif password != password_confirm:
                    st.error("Пароли не совпадают")
                elif len(password) < 4:
                    st.error("Пароль должен быть не менее 4 символов")
                else:
                    success, user_id, message = register_user(username, password)
                    if success:
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        else:
            if st.button("Войти", use_container_width=True):
                if not username or not password:
                    st.error("Заполните все поля")
                else:
                    success, user_id, message = login_user(username, password)
                    if success:
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)


def show_user_info():
    """Отображает информацию о текущем пользователе"""
    with st.sidebar:
        st.markdown("### 👤 Пользователь")
        st.success(f"✅ {st.session_state.username}")

        if st.button("🚪 Выйти", use_container_width=True):
            # Очищаем данные авторизации
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.auth_mode = "login"

            # Очищаем результаты анализов
            st.session_state.current_results = None
            st.session_state.current_eval_ids = None

            # Очищаем историю чата
            st.session_state.messages = [
                {"role": "assistant",
                 "content": "👋 Привет! Я AI-рекрутер. Загрузите файлы резюме и напишите, что нужно сделать: оценить, сравнить или посмотреть статистику."}
            ]

            st.rerun()

        st.markdown("---")


# ============================================
# САЙДБАР
# ============================================

with st.sidebar:
    if st.session_state.user_id is None:
        show_auth_form()
    else:
        show_user_info()

        st.markdown("### 💡 Как использовать")
        st.markdown("""
        **Что можно спросить:**
        - `Оцени` — оценить загруженные файлы
        - `Сравни` — сравнить кандидатов
        - `Покажи статистику`

        **Команды:**
        - `/help` — справка
        - `/clear` — очистить чат
        - `/stats` — статистика
        """)

        st.markdown("---")
        st.markdown("### 🏗️ Архитектура")
        st.markdown("LangGraph: Parser → Scorer → Role Advisor → Formatter")
        st.markdown("---")
        st.markdown("### 📊 Критерии оценки")
        st.markdown("- Hard Skills (35%)\n- Soft Skills (25%)\n- Опыт (25%)\n- Адаптивность (15%)")
        st.markdown("---")

        stats = get_feedback_stats()
        st.metric("💬 Фидбек получено", stats["total"])
        if stats["avg_rating"] > 0:
            st.metric("⭐ Средняя оценка", stats["avg_rating"])

# ============================================
# ЧАТ
# ============================================

# Отображение истории чата
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Загрузка файлов
uploaded_files = st.file_uploader(
    "📎 Загрузите файлы резюме (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="chat_uploader",
    help="Можно загрузить несколько файлов для массовой оценки или сравнения"
)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files


# Функция обработки резюме
def process_resume(file_name: str, raw_text: str) -> dict:
    initial_state: ResumeState = {
        "file_name": file_name,
        "raw_text": raw_text,
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
    }
    result = app_graph.invoke(initial_state)

    # Добавляем текст резюме в результат (для RAG)
    result["raw_text"] = raw_text
    result["cleaned_text"] = result.get("cleaned_text", "")

    return result


# ============================================
# ОБРАБОТКА СООБЩЕНИЙ
# ============================================

if prompt := st.chat_input("Напишите сообщение..."):
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Сбрасываем предыдущие результаты
    st.session_state.current_results = None
    st.session_state.current_eval_ids = None

    with st.spinner("🤔 Думаю..."):
        if prompt.startswith("/"):
            if prompt == "/help":
                response = handle_help()
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt == "/clear":
                st.session_state.messages = [
                    {"role": "assistant", "content": "🧹 Чат очищен! Загрузите новое резюме."}
                ]
                response = "Чат очищен!"
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif prompt == "/stats":
                response = handle_stats(st.session_state.user_id)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = "❌ Неизвестная команда. Напишите /help"
                st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            intent_data = classify_intent(prompt)
            intent = intent_data["intent"]

            if intent == "scoring":
                result_type, output, results, eval_ids = handle_scoring(
                    st.session_state.user_id,
                    intent_data["file_names"],
                    st.session_state.uploaded_files
                )

                st.session_state.messages.append({"role": "assistant", "content": output})

                # Унифицированное сохранение: и single, и batch — в current_results
                if result_type in ["single", "batch"]:
                    st.session_state.current_results = results
                    st.session_state.current_eval_ids = eval_ids

            elif intent == "compare":
                output, results, eval_ids = handle_compare(
                    st.session_state.user_id,
                    intent_data["file_names"],
                    st.session_state.uploaded_files
                )
                st.session_state.messages.append({"role": "assistant", "content": output})
                st.session_state.current_results = results
                st.session_state.current_eval_ids = eval_ids

            elif intent == "stats":
                response = handle_stats(st.session_state.user_id)
                st.session_state.messages.append({"role": "assistant", "content": response})

            elif intent == "help":
                response = handle_help()
                st.session_state.messages.append({"role": "assistant", "content": response})

            else:
                response = handle_chat(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()

# ============================================
# ОТОБРАЖЕНИЕ ДЕТАЛЬНЫХ РЕЗУЛЬТАТОВ И ФИДБЕКА
# ============================================

if st.session_state.current_results:
    st.markdown("---")

    # Кнопка сохранения результатов
    csv_data = export_to_csv(st.session_state.current_results)
    st.download_button(
        label="📥 Скачать результаты (CSV)",
        data=csv_data,
        file_name=f"resumes_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("### 📋 Детальный анализ кандидатов")

    for i, (result, eval_id) in enumerate(zip(st.session_state.current_results, st.session_state.current_eval_ids)):
        with st.expander(f"{i + 1}. {result['file_name']} — {result['total_score']}/100 ({result['recommendation']})"):
            st.markdown(format_detailed_result(result))
            if eval_id:
                # Получаем текст резюме для RAG
                resume_text = result.get("cleaned_text", "") or result.get("raw_text", "")
                render_feedback_form(
                    eval_id,
                    resume_text=resume_text,
                    evaluation_result=result
                )
