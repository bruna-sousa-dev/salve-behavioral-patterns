import pandas as pd
from config import ORIGINAL_DATASET_PATH, CLEANED_TEMPORAL_DATASET_PATH

def load_data(path: str) -> pd.DataFrame:
    """
    Carrega dataset bruto com separador correto.
    """
    df = pd.read_csv(path, sep=",")
    return df

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas de tempo para datetime.
    """
    df["alert_time"] = pd.to_datetime(df["alert_time"], errors="coerce")
    df["served_time"] = pd.to_datetime(df["served_time"], errors="coerce")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove inconsistências básicas.
    """
    # Remove linhas sem alert_time (crítico)
    df = df.dropna(subset=["alert_time"])

    # Remove duplicatas exatas
    df = df.drop_duplicates()

    # Garante tipo booleano
    df["served_flag"] = df["served_flag"].astype(bool)

    return df

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis temporais interpretáveis.
    """

    # Ordenação temporal (CRÍTICO)
    df = df.sort_values("alert_time").reset_index(drop=True)

    # Tempo de resposta (apenas quando atendido)
    df["response_seconds"] = (
        df["served_time"] - df["alert_time"]
    ).dt.total_seconds()

    # Hora e minuto do dia
    df["hour"] = df["alert_time"].dt.hour
    df["minute"] = df["alert_time"].dt.minute
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # Dia da semana
    df["day_of_week"] = df["alert_time"].dt.dayofweek  # 0=segunda
    df["is_weekend"] = df["day_of_week"] >= 5

    # Parte do dia (discretização importante p/ associação)
    def get_part_of_day(hour):
        if 7 <= hour < 9:
            return "early_morning"
        elif 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 14:
            return "lunch"
        elif 14 <= hour < 17:
            return "afternoon"
        else:
            return "off_hours"

    df["part_of_day"] = df["hour"].apply(get_part_of_day)

    return df

def create_event_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria métricas de dinâmica temporal (ESSENCIAL para ML).
    """

    # Tempo desde último alerta
    df["time_since_prev_alert_s"] = (
        df["alert_time"].diff().dt.total_seconds()
    )

    # Preenche primeiro valor
    df["time_since_prev_alert_s"] = df["time_since_prev_alert_s"].fillna(0)

    return df

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria volume de alertas em janelas móveis.
    """

    df = df.set_index("alert_time")

    # Contagem de alertas em janelas temporais
    df["alerts_last_15m"] = df["served_flag"].rolling("15min").count()
    df["alerts_last_60m"] = df["served_flag"].rolling("60min").count()

    df = df.reset_index()

    return df

def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seleciona colunas finais organizadas.
    """

    columns = [
        "alert_time",
        "served_time",
        "served_flag",
        "response_seconds",
        "hour",
        "minute",
        "minute_of_day",
        "day_of_week",
        "is_weekend",
        "part_of_day",
        "time_since_prev_alert_s",
        "alerts_last_15m",
        "alerts_last_60m",
    ]

    return df[columns]

# =========================================================
# DESIGN DECISION:
# Pipeline determinístico de limpeza e enriquecimento temporal
# - Evita data leakage (não usa served_time como feature explicativa direta)
# - Gera variáveis interpretáveis (importante para regras de associação)
# - Mantém consistência temporal (ordenação + deltas)
# =========================================================
def deterministic_pipeline():
    print("Carregando dados...")
    df = load_data(ORIGINAL_DATASET_PATH)

    print("Parse de datas...")
    df = parse_datetime(df)

    print("Limpeza...")
    df = clean_data(df)

    print("Features temporais...")
    df = create_temporal_features(df)

    print("Dinâmica de eventos...")
    df = create_event_dynamics(df)

    print("Features rolling...")
    df = create_rolling_features(df)

    print("Finalizando dataset...")
    df = finalize_dataset(df)

    print("Salvando...")
    df.to_csv(CLEANED_TEMPORAL_DATASET_PATH, sep=",", index=False)

    print("Dataset limpo gerado com sucesso!")
    print(df.head())
