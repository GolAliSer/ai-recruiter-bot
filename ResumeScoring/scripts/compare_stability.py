#!/usr/bin/env python
"""
Сравнение стабильности двух прогонов метрик
Запускать: python scripts/compare_stability.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Настройка стиля
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(file_path: str) -> pd.DataFrame:
    """Загружает результаты тестирования"""
    df = pd.read_csv(file_path)
    # Добавляем ID для идентификации резюме
    df['id'] = range(1, len(df) + 1)
    return df


def compare_runs(run1_path: str, run2_path: str, output_dir: str = "stability_analysis"):
    """Сравнивает два прогона и строит графики"""

    # Создаём папку для результатов
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем данные
    df1 = load_results(run1_path)
    df2 = load_results(run2_path)

    print("=" * 60)
    print("📊 СРАВНЕНИЕ СТАБИЛЬНОСТИ ОЦЕНОК")
    print("=" * 60)
    print(f"Прогон 1: {run1_path}")
    print(f"  • Резюме: {len(df1)}")
    print(f"  • Средний балл: {df1['total_score'].mean():.1f}")
    print(f"  • Медиана: {df1['total_score'].median():.1f}")
    print(f"  • Стандартное отклонение: {df1['total_score'].std():.2f}")
    print()
    print(f"Прогон 2: {run2_path}")
    print(f"  • Резюме: {len(df2)}")
    print(f"  • Средний балл: {df2['total_score'].mean():.1f}")
    print(f"  • Медиана: {df2['total_score'].median():.1f}")
    print(f"  • Стандартное отклонение: {df2['total_score'].std():.2f}")

    # Проверяем, что количество резюме совпадает
    if len(df1) != len(df2):
        print(f"\n⚠️ Внимание: количество резюме разное ({len(df1)} vs {len(df2)})")
        min_len = min(len(df1), len(df2))
        df1 = df1.head(min_len)
        df2 = df2.head(min_len)

    # Вычисляем разницу
    df1 = df1.sort_values('id')
    df2 = df2.sort_values('id')

    diff = df2['total_score'].values - df1['total_score'].values
    abs_diff = np.abs(diff)

    print("\n" + "=" * 60)
    print("📊 АНАЛИЗ СТАБИЛЬНОСТИ")
    print("=" * 60)
    print(f"\n• Средняя абсолютная разница: {abs_diff.mean():.2f} балла")
    print(f"• Медиана абсолютной разницы: {np.median(abs_diff):.2f} балла")
    print(f"• Стандартное отклонение разницы: {diff.std():.2f}")
    print(f"• Минимальная разница: {diff.min():.2f}")
    print(f"• Максимальная разница: {diff.max():.2f}")

    # Процент резюме с разницей ≤5 баллов
    within_5 = (abs_diff <= 5).sum()
    print(f"\n• Резюме с разницей ≤5 баллов: {within_5}/{len(df1)} ({within_5 / len(df1) * 100:.1f}%)")

    # Процент резюме с разницей ≤10 баллов
    within_10 = (abs_diff <= 10).sum()
    print(f"• Резюме с разницей ≤10 баллов: {within_10}/{len(df1)} ({within_10 / len(df1) * 100:.1f}%)")

    # ============================================
    # ГРАФИК 1: Сравнение баллов (scatter plot)
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # Точки
    ax.scatter(df1['total_score'], df2['total_score'], alpha=0.7, s=80, color='#66b3ff', edgecolor='black')

    # Линия идеального совпадения (y=x)
    max_val = max(df1['total_score'].max(), df2['total_score'].max())
    min_val = min(df1['total_score'].min(), df2['total_score'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное совпадение (y=x)')

    # Линия тренда
    z = np.polyfit(df1['total_score'], df2['total_score'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min_val, max_val, 100)
    ax.plot(x_trend, p(x_trend), 'g-', linewidth=2, label=f'Тренд: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.set_xlabel('Прогон 1 (баллы)', fontsize=12)
    ax.set_ylabel('Прогон 2 (баллы)', fontsize=12)
    ax.set_title('Сравнение баллов двух прогонов', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_scatter.png", dpi=150)
    plt.close()
    print(f"\n✅ График 1 сохранён: {output_dir}/stability_scatter.png")

    # ============================================
    # ГРАФИК 2: Гистограмма разницы баллов
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(diff, bins=20, edgecolor='black', alpha=0.7, color='#ffcc99')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Нулевая разница')
    ax.axvline(diff.mean(), color='green', linestyle='--', linewidth=2, label=f'Среднее: {diff.mean():.2f}')

    ax.set_xlabel('Разница (Прогон 2 - Прогон 1)', fontsize=12)
    ax.set_ylabel('Количество резюме', fontsize=12)
    ax.set_title('Распределение разницы баллов между прогонами', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_histogram.png", dpi=150)
    plt.close()
    print(f"✅ График 2 сохранён: {output_dir}/stability_histogram.png")

    # ============================================
    # ГРАФИК 3: Boxplot сравнения
    # ============================================
    fig, ax = plt.subplots(figsize=(8, 6))

    data_to_plot = [df1['total_score'], df2['total_score']]
    bp = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True, tick_labels=['Прогон 1', 'Прогон 2'])

    colors = ['#66b3ff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Баллы', fontsize=12)
    ax.set_title('Сравнение распределения баллов', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_boxplot.png", dpi=150)
    plt.close()
    print(f"✅ График 3 сохранён: {output_dir}/stability_boxplot.png")

    # ============================================
    # ГРАФИК 4: Порезюмное сравнение
    # ============================================
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(df1))
    width = 0.35

    ax.bar(x - width / 2, df1['total_score'], width, label='Прогон 1', color='#66b3ff', edgecolor='black')
    ax.bar(x + width / 2, df2['total_score'], width, label='Прогон 2', color='#ff9999', edgecolor='black')

    ax.set_xlabel('Номер резюме', fontsize=12)
    ax.set_ylabel('Баллы', fontsize=12)
    ax.set_title('Порезюмное сравнение оценок', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_by_resume.png", dpi=150)
    plt.close()
    print(f"✅ График 4 сохранён: {output_dir}/stability_by_resume.png")

    # ============================================
    # Сохраняем таблицу сравнения
    # ============================================
    comparison_df = pd.DataFrame({
        'id': df1['id'],
        'file_name': df1['file_name'],
        'Прогон_1': df1['total_score'],
        'Прогон_2': df2['total_score'],
        'Разница': diff,
        '|Разница|': abs_diff
    })
    comparison_df.to_csv(f"{output_dir}/comparison_table.csv", index=False, encoding='utf-8-sig')
    print(f"✅ Таблица сравнения сохранена: {output_dir}/comparison_table.csv")

    # Сохраняем сводную статистику
    summary = {
        "Показатель": [
            "Средний балл (Прогон 1)",
            "Средний балл (Прогон 2)",
            "Медиана (Прогон 1)",
            "Медиана (Прогон 2)",
            "Средняя абсолютная разница",
            "Медиана абсолютной разницы",
            "Максимальная разница",
            "Резюме с разницей ≤5 баллов (%)",
            "Резюме с разницей ≤10 баллов (%)",
            "Корреляция Пирсона"
        ],
        "Значение": [
            f"{df1['total_score'].mean():.1f}",
            f"{df2['total_score'].mean():.1f}",
            f"{df1['total_score'].median():.1f}",
            f"{df2['total_score'].median():.1f}",
            f"{abs_diff.mean():.2f}",
            f"{np.median(abs_diff):.2f}",
            f"{diff.max():.2f}",
            f"{within_5 / len(df1) * 100:.1f}%",
            f"{within_10 / len(df1) * 100:.1f}%",
            f"{df1['total_score'].corr(df2['total_score']):.3f}"
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/stability_summary.csv", index=False, encoding='utf-8-sig')
    print(f"✅ Сводная статистика сохранена: {output_dir}/stability_summary.csv")

    print("\n" + "=" * 60)
    print("✅ АНАЛИЗ ЗАВЕРШЁН!")
    print(f"📁 Результаты в папке: {output_dir}/")
    print("=" * 60)

    return comparison_df, summary_df


def main():
    """Основная функция"""

    # Укажите пути к вашим файлам
    # Пример: "metrics_results/test_results_20250421_120000.csv"

    print("📁 Введите пути к файлам результатов:")
    print("-" * 50)

    run1 = input("Путь к первому прогону: ").strip()
    run2 = input("Путь ко второму прогону: ").strip()

    if not os.path.exists(run1):
        print(f"\n❌ Файл не найден: {run1}")
        return

    if not os.path.exists(run2):
        print(f"\n❌ Файл не найден: {run2}")
        return

    compare_runs(run1, run2)


if __name__ == "__main__":
    main()