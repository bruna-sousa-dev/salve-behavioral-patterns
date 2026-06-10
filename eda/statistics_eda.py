import pandas as pd
import numpy as np
from config import OBSERVATIONAL_CLEANED_PREPROCESSED_DATASET_PATH


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    return df


df = load_data(OBSERVATIONAL_CLEANED_PREPROCESSED_DATASET_PATH)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def response_time_statistics():
    print_section("1. Estatísticas do Tempo de Resposta")

    response = df["response_seconds"].dropna()

    stats = {
        "n_atendidos": response.count(),
        "media_s": response.mean(),
        "mediana_s": response.median(),
        "moda_s": response.mode().iloc[0] if not response.mode().empty else np.nan,
        "desvio_padrao_s": response.std(),
        "variancia_s2": response.var(),
        "min_s": response.min(),
        "q1_25_s": response.quantile(0.25),
        "q2_50_s": response.quantile(0.50),
        "q3_75_s": response.quantile(0.75),
        "iqr_s": response.quantile(0.75) - response.quantile(0.25),
        "p90_s": response.quantile(0.90),
        "p95_s": response.quantile(0.95),
        "max_s": response.max(),
        "assimetria_skewness": response.skew(),
        "curtose_kurtosis": response.kurtosis()
    }

    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    return stats


def boxplot_outlier_statistics():
    print_section("2. Estatísticas para Boxplot e Outliers")

    response = df["response_seconds"].dropna()

    q1 = response.quantile(0.25)
    q3 = response.quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = response[(response < lower_limit) | (response > upper_limit)]

    print(f"Q1: {q1:.4f}")
    print(f"Q3: {q3:.4f}")
    print(f"IQR: {iqr:.4f}")
    print(f"Limite inferior: {lower_limit:.4f}")
    print(f"Limite superior: {upper_limit:.4f}")
    print(f"Número de outliers: {outliers.count()}")

    if not outliers.empty:
        print("\nOutliers identificados:")
        print(outliers.to_string(index=True))

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "outliers": outliers
    }


def service_rate_by_hour():
    print_section("3. Taxa de Atendimento por Hora")

    result = (
        df.groupby("hour")
        .agg(
            total_alerts=("served_flag", "count"),
            served_alerts=("served_flag", "sum"),
            service_rate=("served_flag", "mean")
        )
        .reset_index()
    )

    result["service_rate_percent"] = result["service_rate"] * 100

    print(result.to_string(index=False))

    return result


def service_rate_weekday_vs_weekend():
    print_section("4. Taxa de Atendimento: Dia Útil vs Fim de Semana")

    result = (
        df.groupby("is_weekend")
        .agg(
            total_alerts=("served_flag", "count"),
            served_alerts=("served_flag", "sum"),
            service_rate=("served_flag", "mean")
        )
        .reset_index()
    )

    result["period"] = result["is_weekend"].map({
        False: "Weekdays",
        True: "Weekend"
    })

    result["service_rate_percent"] = result["service_rate"] * 100

    result = result[[
        "period",
        "is_weekend",
        "total_alerts",
        "served_alerts",
        "service_rate",
        "service_rate_percent"
    ]]

    print(result.to_string(index=False))

    return result


def correlation_statistics():
    print_section("5. Matriz de Correlação")

    numeric_cols = [
        "response_seconds",
        "time_since_prev_alert_s",
        "alerts_last_15m",
        "alerts_last_60m",
        "hour"
    ]

    corr = df[numeric_cols].corr()

    print(corr.round(4).to_string())

    return corr


def dataset_quality_summary():
    print_section("6. Resumo de Qualidade dos Dados")

    total_records = len(df)
    served_count = df["served_flag"].sum()
    not_served_count = total_records - served_count

    print(f"Total de registros: {total_records}")
    print(f"Alertas atendidos: {served_count}")
    print(f"Alertas não atendidos: {not_served_count}")
    print(f"Taxa geral de atendimento: {(served_count / total_records) * 100:.2f}%")

    print("\nValores ausentes por coluna:")
    print(df.isna().sum().to_string())

    print(f"\nDuplicatas exatas: {df.duplicated().sum()}")

    if "alert_time" in df.columns:
        df["alert_time"] = pd.to_datetime(df["alert_time"], errors="coerce")
        print(f"alert_time inválidos: {df['alert_time'].isna().sum()}")
        print(f"Período inicial: {df['alert_time'].min()}")
        print(f"Período final: {df['alert_time'].max()}")

    if "served_time" in df.columns:
        df["served_time"] = pd.to_datetime(df["served_time"], errors="coerce")
        print(f"served_time ausentes/inválidos: {df['served_time'].isna().sum()}")

    return {
        "total_records": total_records,
        "served_count": served_count,
        "not_served_count": not_served_count
    }


def run_all_statistics():
    dataset_quality_summary()
    response_time_statistics()
    boxplot_outlier_statistics()
    service_rate_by_hour()
    service_rate_weekday_vs_weekend()
    correlation_statistics()


if __name__ == "__main__":
    run_all_statistics()
