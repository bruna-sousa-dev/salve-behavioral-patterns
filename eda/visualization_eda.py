import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import CLEANED_TEMPORAL_DATASET_PATH

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    return df


df = load_data(CLEANED_TEMPORAL_DATASET_PATH)


def visualize_response_time_distribution():
    plt.figure(figsize=(8, 5))
    sns.histplot(df["response_seconds"].dropna(), bins=10, kde=True)
    # plt.title("Response Time Distribution")
    plt.xlabel("Seconds")
    plt.xticks(np.arange(0, df["response_seconds"].max(), 20))
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def visualize_response_time_boxplot():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["response_seconds"])
    # plt.title("Response Time Boxplot")
    plt.xlabel("Seconds")
    plt.xticks(np.arange(0, df["response_seconds"].max(), 20))
    plt.tight_layout()
    plt.show()


def visualize_response_time_by_hour():
    plt.figure(figsize=(8, 5))
    df.groupby("hour")["served_flag"].mean().plot(kind="bar")
    # plt.title("Hourly Service Rate")
    plt.xlabel("Hours of the day")
    plt.ylabel("Proportion")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.show()


def visualize_weekend_vs_weekday_response_rate():
    plt.figure(figsize=(8, 5))

    data = df.groupby("is_weekend")["served_flag"].mean()
    data.index = data.index.map({False: "Weekdays", True: "Weekend"})
    data.plot(kind="bar")
    # plt.title("Service Rate: Weekday vs. Weekend")
    plt.xlabel("")
    plt.ylabel("Proportion")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.show()


def visualize_correlation_matrix():
    plt.figure(figsize=(8, 5))
    numeric_cols = [
        "response_seconds",
        "time_since_prev_alert_s",
        "alerts_last_15m",
        "alerts_last_60m",
        "hour"
    ]
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    # plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def visualize_all():
    visualize_response_time_distribution()
    visualize_response_time_boxplot()
    visualize_response_time_by_hour()
    visualize_weekend_vs_weekday_response_rate()
    visualize_correlation_matrix()
