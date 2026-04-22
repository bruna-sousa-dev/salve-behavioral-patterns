import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from config import CLEANED_TEMPORAL_DATASET_PATH


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    return df


df = load_data(CLEANED_TEMPORAL_DATASET_PATH)


def visualize_response_time_distribution():
    plt.figure(figsize=(8, 5))
    sns.histplot(df["response_seconds"].dropna(), bins=10, kde=True)
    plt.title("Distribuição do Tempo de Resposta")
    plt.xlabel("Segundos")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()


def visualize_response_time_boxplot():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["response_seconds"])
    plt.title("Boxplot do Tempo de Resposta")
    plt.tight_layout()
    plt.show()


def visualize_response_time_by_hour():
    plt.figure(figsize=(8, 5))
    df.groupby("hour")["served_flag"].mean().plot(kind="bar")
    plt.title("Taxa de Atendimento por Hora")
    plt.ylabel("Proporção")
    plt.tight_layout()
    plt.show()


def visualize_weekend_vs_weekday_response_rate():
    plt.figure(figsize=(8, 5))
    df.groupby("is_weekend")["served_flag"].mean().plot(kind="bar")
    plt.title("Taxa de Atendimento: Dia Útil vs Fim de Semana")
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
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.show()


def visualize_all():
    visualize_response_time_distribution()
    visualize_response_time_boxplot()
    visualize_response_time_by_hour()
    visualize_weekend_vs_weekday_response_rate()
    visualize_correlation_matrix()
