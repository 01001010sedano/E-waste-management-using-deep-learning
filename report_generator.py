# report_generator.py

import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # ✅ Add this line to avoid RuntimeError in background threads
import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_toxicity_report_graph(json_path="summary.json", save_path="toxicity_report.png"):
    """Pie chart showing overall toxicity distribution"""
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[ERROR] Could not read {json_path}.")
        return

    toxicity_levels = [entry.get("toxicity_level") for entry in data if "toxicity_level" in entry]

    if not toxicity_levels:
        print("[INFO] No data to plot.")
        return

    toxicity_count = Counter(toxicity_levels)

    labels = list(toxicity_count.keys())
    sizes = list(toxicity_count.values())

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Toxicity Distribution of E-Waste")
    plt.axis("equal")
    plt.savefig(save_path)
    plt.close()

    print(f"[SUCCESS] Pie chart saved as {save_path}")


def generate_bar_graph(json_path="summary.json", save_path="toxicity_bar_chart.png"):
    """Bar chart showing count per toxicity level"""
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[ERROR] Could not read {json_path}.")
        return

    toxicity_levels = [entry.get("toxicity_level") for entry in data if "toxicity_level" in entry]
    if not toxicity_levels:
        print("[INFO] No data found to generate graph.")
        return

    counter = Counter(toxicity_levels)

    labels = list(counter.keys())
    values = list(counter.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color="skyblue")
    plt.title("Total Count per Toxicity Level")
    plt.xlabel("Toxicity Level")
    plt.ylabel("Count")
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{int(height)}', ha='center', va='bottom')

    plt.savefig(save_path)
    plt.close()

    print(f"[SUCCESS] Bar chart saved as {save_path}")


def generate_line_graph_by_date(json_path="summary.json", save_path="toxicity_line_chart.png"):
    """Line graph showing daily toxicity level counts"""
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[ERROR] Could not read {json_path}.")
        return

    df = pd.DataFrame(data)
    if 'timestamp' not in df.columns or 'toxicity_level' not in df.columns:
        print("[INFO] Not enough data to generate line graph.")
        return

    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    toxicity_counts = df.groupby(['date', 'toxicity_level']).size().unstack(fill_value=0)

    toxicity_counts.plot(kind='line', marker='o', figsize=(10, 6))
    plt.title("Daily E-Waste Toxicity Trend")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[SUCCESS] Line chart saved as {save_path}")
