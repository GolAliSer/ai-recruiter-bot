"""Узел 4 LangGraph: Форматирование вывода для пользователя"""

from src.state import ResumeState


def formatter_node(state: ResumeState) -> ResumeState:
    """Форматирование вывода для пользователя"""

    # Безопасное получение значений
    s = state.get("scores", {})
    role_confidence = state.get("role_confidence", 50)
    role_reasoning = state.get("role_reasoning", "Анализ выполнен")
    total_score = state.get("total_score", 0)
    recommendation = state.get("recommendation", "Consider")
    detected_role = state.get("detected_role", "Не определена")
    file_name = state.get("file_name", "Неизвестный файл")
    word_count = state.get("word_count", 0)
    explanation = state.get("explanation", "Нет объяснения")

    rec_emoji = {
        "Strong Hire": "🟢",
        "Hire": "🟡",
        "Consider": "🟠",
        "Pass": "🔴"
    }.get(recommendation, "⚪")

    def progress_bar(value, max_val, width=15):
        filled = int(width * value / max_val) if max_val > 0 else 0
        return "█" * filled + "░" * (width - filled)

    # Визуализация уверенности
    confidence_filled = int(role_confidence / 10)
    confidence_bar = "█" * confidence_filled + "░" * (10 - confidence_filled)

    output = f"""
**📊 РЕЗУЛЬТАТ ОЦЕНКИ**

**Кандидат:** `{file_name}`
**Предполагаемая роль:** {detected_role}
**Уверенность:** {role_confidence}% [{confidence_bar}]
**Обоснование:** {role_reasoning}
**Объем текста:** {word_count} слов

**🏆 ОБЩИЙ БАЛЛ:** **{total_score}/100**
**🎯 ВЕРДИКТ:** {rec_emoji} **{recommendation}**

---

**📈 ДЕТАЛИЗАЦИЯ:**

• **Hard Skills** (35%): {s.get('hard_skills', 0)}/35 {progress_bar(s.get('hard_skills', 0), 35)}
• **Soft Skills** (25%): {s.get('soft_skills', 0)}/25 {progress_bar(s.get('soft_skills', 0), 25)}
• **Опыт** (25%): {s.get('experience', 0)}/25 {progress_bar(s.get('experience', 0), 25)}
• **Адаптивность** (15%): {s.get('adaptability', 0)}/15 {progress_bar(s.get('adaptability', 0), 15)}

---

**💬 {explanation}**
"""
    state["final_output"] = output
    return state
