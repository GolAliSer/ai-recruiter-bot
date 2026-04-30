#!/usr/bin/env python
"""
Скрипт для сбора метрик работы системы на тестовой выборке.
Запускать отдельно: python scripts/collect_metrics.py
"""

import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Добавляем корневую папку в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import app_graph
from src.state import ResumeState
from src.config import MODEL_NAME

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Настройка паузы между запросами (секунды)
REQUEST_DELAY = 30

def process_resume(file_name: str, raw_text: str) -> dict:
    """Запускает граф обработки резюме"""
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
    start_time = time.time()
    result = app_graph.invoke(initial_state)
    elapsed_time = time.time() - start_time

    return {
        "file_name": file_name,
        "total_score": result["total_score"],
        "recommendation": result["recommendation"],
        "detected_role": result["detected_role"],
        "word_count": result["word_count"],
        "hard_skills": result["scores"]["hard_skills"],
        "soft_skills": result["scores"]["soft_skills"],
        "experience": result["scores"]["experience"],
        "adaptability": result["scores"]["adaptability"],
        "response_time": round(elapsed_time, 2),
        "explanation": result["explanation"][:100]
    }


def load_test_resumes(csv_path: str, limit: int = 100) -> list:
    """Загружает тестовые резюме из CSV"""
    df = pd.read_csv(csv_path)

    # Берём только указанное количество
    df = df.head(limit)

    resumes = []
    for idx, row in df.iterrows():
        # Собираем текст резюме из доступных полей
        text_parts = []
        if pd.notna(row.get('experience_text')):
            text_parts.append(str(row['experience_text']))
        if pd.notna(row.get('title')):
            text_parts.append(str(row['title']))

        resume_text = "\n".join(text_parts)

        if len(resume_text) > 100:
            resumes.append({
                "id": idx,
                "name": f"resume_{idx}.txt",
                "text": resume_text,
                "true_role": row.get('specialization', 'unknown')
            })

    return resumes


def collect_metrics(resumes: list, delay: float = REQUEST_DELAY) -> pd.DataFrame:
    """Собирает метрики для всех резюме с паузой между запросами"""
    results = []
    total = len(resumes)

    print(f"\n📊 Начинаем анализ {total} резюме...")
    print(f"🔄 Модель: {MODEL_NAME}")
    print(f"⏱️  Пауза между запросами: {delay} сек")
    print("-" * 50)

    for i, resume in enumerate(resumes):
        request_start = time.time()

        print(f"  [{i + 1}/{total}] Анализ: {resume['name']}...", end=" ", flush=True)

        try:
            result = process_resume(resume["name"], resume["text"])
            results.append(result)
            print(f"✅ {result['total_score']}/100 ({result['response_time']}с)")
        except Exception as e:
            print(f"❌ Ошибка: {str(e)[:50]}")
            results.append({
                "file_name": resume["name"],
                "total_score": 0,
                "recommendation": "Error",
                "detected_role": "error",
                "word_count": 0,
                "hard_skills": 0,
                "soft_skills": 0,
                "experience": 0,
                "adaptability": 0,
                "response_time": 0,
                "explanation": f"Ошибка: {str(e)[:100]}"
            })

        # Добавляем паузу между запросами (кроме последнего)
        elapsed = time.time() - request_start
        wait_time = max(0, delay - elapsed)

        if i < total - 1 and wait_time > 0:
            print(f"  ⏳ Пауза {wait_time:.1f}с...")
            time.sleep(wait_time)

    return pd.DataFrame(results)


def plot_distribution(df: pd.DataFrame, output_dir: str):
    """Строит график распределения баллов"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Гистограмма
    ax.hist(df['total_score'], bins=20, edgecolor='black', alpha=0.7, color='#66b3ff')

    # Среднее и медиана
    mean_score = df['total_score'].mean()
    median_score = df['total_score'].median()

    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_score:.1f}')
    ax.axvline(median_score, color='orange', linestyle='--', linewidth=2, label=f'Медиана: {median_score:.1f}')

    ax.set_xlabel('Общий балл', fontsize=12)
    ax.set_ylabel('Количество резюме', fontsize=12)
    ax.set_title(f'Распределение баллов оценки резюме (N={len(df)})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distribution.png", dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/score_distribution.png")


def plot_by_role(df: pd.DataFrame, output_dir: str):
    """Строит график баллов по ролям"""
    # Группировка по ролям
    role_stats = df.groupby('detected_role').agg({
        'total_score': ['mean', 'count', 'std']
    }).round(1)
    role_stats.columns = ['Средний балл', 'Кол-во', 'Std']
    role_stats = role_stats.sort_values('Средний балл', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(role_stats.index, role_stats['Средний балл'], edgecolor='black', color='#99ff99')
    ax.bar_label(bars, fmt='%.0f', padding=3)

    ax.set_xlabel('Предполагаемая роль', fontsize=12)
    ax.set_ylabel('Средний балл', fontsize=12)
    ax.set_title('Средний балл по типам ролей', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_by_role.png", dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/score_by_role.png")

    return role_stats


def plot_response_time(df: pd.DataFrame, output_dir: str):
    """Строит график времени ответа"""
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_times = df[df['response_time'] > 0]['response_time']

    ax.hist(valid_times, bins=15, edgecolor='black', alpha=0.7, color='#ffcc99')
    ax.axvline(valid_times.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Среднее: {valid_times.mean():.2f}с')
    ax.axvline(valid_times.median(), color='orange', linestyle='--', linewidth=2,
               label=f'Медиана: {valid_times.median():.2f}с')

    ax.set_xlabel('Время ответа (секунды)', fontsize=12)
    ax.set_ylabel('Количество запросов', fontsize=12)
    ax.set_title('Распределение времени ответа LLM', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_time.png", dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/response_time.png")


def plot_criteria_radar(df: pd.DataFrame, output_dir: str):
    """Строит лепестковую диаграмму по критериям"""
    criteria_avg = {
        'Hard Skills': df['hard_skills'].mean(),
        'Soft Skills': df['soft_skills'].mean(),
        'Опыт': df['experience'].mean(),
        'Адаптивность': df['adaptability'].mean()
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    categories = list(criteria_avg.keys())
    values = list(criteria_avg.values())
    # Нормализуем к максимальным значениям
    max_values = [35, 25, 25, 15]
    normalized = [v / m for v, m in zip(values, max_values)]

    angles = [n / len(categories) * 2 * 3.14159 for n in range(len(categories))]
    angles += angles[:1]
    normalized += normalized[:1]

    ax.plot(angles, normalized, 'o-', linewidth=2, color='#66b3ff')
    ax.fill(angles, normalized, alpha=0.25, color='#66b3ff')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Средние оценки по критериям (нормализовано)', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/criteria_radar.png", dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/criteria_radar.png")


def plot_criteria_boxplot(df: pd.DataFrame, output_dir: str):
    """Строит boxplot для каждого критерия"""
    fig, ax = plt.subplots(figsize=(10, 6))

    criteria_data = [
        df['hard_skills'],
        df['soft_skills'],
        df['experience'],
        df['adaptability']
    ]

    bp = ax.boxplot(criteria_data, patch_artist=True, showmeans=True)

    # Настройка цветов
    colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(['Hard Skills', 'Soft Skills', 'Опыт', 'Адаптивность'])
    ax.set_ylabel('Баллы', fontsize=12)
    ax.set_title('Распределение оценок по критериям', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/criteria_boxplot.png", dpi=150)
    plt.close()
    print(f"✅ График сохранён: {output_dir}/criteria_boxplot.png")


def main():
    """Основная функция"""
    print("=" * 60)
    print("📊 СБОР МЕТРИК ДЛЯ ВКР")
    print("=" * 60)

    # Создаём папку для результатов
    output_dir = "metrics_results"
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем тестовые резюме
    csv_path = "data/test_resumes.csv"

    if not os.path.exists(csv_path):
        print(f"❌ Файл {csv_path} не найден!")
        print("   Убедитесь, что тестовый датасет находится в папке data/")
        return

    print(f"📂 Загрузка резюме из: {csv_path}")

    # Можно изменить limit здесь для тестирования
    resumes = load_test_resumes(csv_path, limit=50)
    print(f"✅ Загружено {len(resumes)} резюме для анализа")

    # Собираем метрики с паузой между запросами
    df = collect_metrics(resumes, delay=REQUEST_DELAY)

    # Сохраняем результаты в CSV
    csv_output = f"{output_dir}/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_output, index=False, encoding='utf-8-sig')
    print(f"\n💾 Результаты сохранены: {csv_output}")

    # Фильтруем только успешные результаты (не ошибки)
    valid_df = df[df['total_score'] > 0].copy()

    if len(valid_df) == 0:
        print("\n❌ Нет успешных результатов для анализа!")
        return

    # Статистика
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА")
    print("=" * 60)

    print(f"\n📈 Общая статистика по баллам:")
    print(f"   • Средний балл: {valid_df['total_score'].mean():.1f}/100")
    print(f"   • Медиана: {valid_df['total_score'].median():.1f}")
    print(f"   • Минимум: {valid_df['total_score'].min():.1f}")
    print(f"   • Максимум: {valid_df['total_score'].max():.1f}")
    print(f"   • Стандартное отклонение: {valid_df['total_score'].std():.2f}")

    print(f"\n⏱️ Время ответа:")
    valid_times = valid_df[valid_df['response_time'] > 0]['response_time']
    if len(valid_times) > 0:
        print(f"   • Среднее время: {valid_times.mean():.2f} сек")
        print(f"   • Медиана: {valid_times.median():.2f} сек")
        print(f"   • Минимум: {valid_times.min():.2f} сек")
        print(f"   • Максимум: {valid_times.max():.2f} сек")

    print(f"\n🎯 Распределение рекомендаций:")
    rec_counts = valid_df['recommendation'].value_counts()
    for rec, count in rec_counts.items():
        print(f"   • {rec}: {count} ({count / len(valid_df) * 100:.1f}%)")

    # Строим графики
    print("\n" + "=" * 60)
    print("📊 ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 60)

    plot_distribution(valid_df, output_dir)
    role_stats = plot_by_role(valid_df, output_dir)
    plot_response_time(valid_df, output_dir)
    plot_criteria_radar(valid_df, output_dir)
    plot_criteria_boxplot(valid_df, output_dir)

    # Сохраняем статистику по ролям
    role_stats.to_csv(f"{output_dir}/role_statistics.csv")

    # Сохраняем общую статистику в файл
    stats_summary = {
        "Дата сбора": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Модель": MODEL_NAME,
        "Количество резюме": len(valid_df),
        "Средний балл": round(valid_df['total_score'].mean(), 1),
        "Медиана": round(valid_df['total_score'].median(), 1),
        "Стандартное отклонение": round(valid_df['total_score'].std(), 2),
        "Среднее время ответа (сек)": round(valid_times.mean(), 2) if len(valid_times) > 0 else 0,
        "Медиана времени ответа (сек)": round(valid_times.median(), 2) if len(valid_times) > 0 else 0,
    }

    stats_df = pd.DataFrame([stats_summary])
    stats_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("✅ МЕТРИКИ СОБРАНЫ!")
    print(f"📁 Результаты в папке: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()