"""Узел 4 LangGraph: Форматирование вывода для пользователя"""

from src.state import ResumeState


def formatter_node(state: ResumeState) -> ResumeState:
    """Форматирование вывода для пользователя"""
    s = state["scores"]

    rec_emoji = {
        "Strong Hire": "🟢",
        "Hire": "🟡",
        "Consider": "🟠",
        "Pass": "🔴"
    }.get(state["recommendation"], "⚪")

    def progress_bar(value, max_val, width=15):
        filled = int(width * value / max_val)
        return "█" * filled + "░" * (width - filled)

    # Визуализация уверенности
    confidence_bar = "█" * int(state.get("role_confidence", 50) / 10) + "░" * (
                10 - int(state.get("role_confidence", 50) / 10))

    output = f"""
**📊 РЕЗУЛЬТАТ ОЦЕНКИ**

**Кандидат:** `{state['file_name']}`
**Предполагаемая роль:** {state['detected_role']}
**Уверенность:** {state.get('role_confidence', 50)}% [{confidence_bar}]
**Обоснование:** {state.get('role_reasoning', 'Анализ выполнен')}
**Объем текста:** {state['word_count']} слов

**🏆 ОБЩИЙ БАЛЛ:** **{state['total_score']}/100**
**🎯 ВЕРДИКТ:** {rec_emoji} **{state['recommendation']}**

---

**📈 ДЕТАЛИЗАЦИЯ:**

• **Hard Skills** (35%): {s['hard_skills']}/35 {progress_bar(s['hard_skills'], 35)}
• **Soft Skills** (25%): {s['soft_skills']}/25 {progress_bar(s['soft_skills'], 25)}
• **Опыт** (25%): {s['experience']}/25 {progress_bar(s['experience'], 25)}
• **Адаптивность** (15%): {s['adaptability']}/15 {progress_bar(s['adaptability'], 15)}

---

**💬 {state['explanation']}**
"""
    state["final_output"] = output
    return state