import pandas as pd
import numpy as np
from config import (
    OBSERVATIONAL_FEATURE_ENGINEERING_DATASET_PATH, 
    SYNTETIC_FEATURE_ENGINEERING_DATASET_PATH,
    OBSERVATIONAL_DISCRETIZATION_CATEGORICAL_DATASET_PATH,
    SYNTETIC_DISCRETIZATION_CATEGORICAL_DATASET_PATH
)

# **************************************************************************************
# Pipeline para discretização dos dados

def load_data(path: str) -> pd.DataFrame:
    """
    Carrega a base limpa com separador ','.
    """
    df = pd.read_csv(path, sep=",")
    return df

# Discretização e criação de variáveis categóricas para regras de associação

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas temporais para datetime.
    """
    df["alert_time"] = pd.to_datetime(df["alert_time"], errors="coerce")
    df["served_time"] = pd.to_datetime(df["served_time"], errors="coerce")
    return df

def create_hour_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria faixas horárias discretas.
    """
    def hour_bin(hour: int) -> str:
        if 7 <= hour < 9:
            return "07_08"
        elif 9 <= hour < 11:
            return "09_10"
        elif 11 <= hour < 13:
            return "11_12"
        elif 13 <= hour < 15:
            return "13_14"
        elif 15 <= hour < 17:
            return "15_16"
        else:
            return "other"

    df["hour_bin"] = df["hour"].apply(hour_bin)
    return df

def classify_response_time(value: float) -> str:
    """
    Classifica tempo de resposta.
    """
    if pd.isna(value):
        return "not_served"
    elif value <= 60:
        return "fast_le_60s"
    elif value <= 180:
        return "normal_61_180s"
    else:
        return "slow_gt_180s"

def classify_time_since_prev_alert(value: float) -> str:
    """
    Classifica espaçamento entre alertas.
    """
    if pd.isna(value):
        return "unknown"
    elif value <= 300:
        return "short_le_5m"
    elif value <= 1800:
        return "medium_5m_30m"
    else:
        return "long_gt_30m"

def classify_alert_volume_15m(value: float) -> str:
    """
    Classifica volume recente de alertas em 15 min.
    """
    if pd.isna(value):
        return "unknown"
    elif value <= 1:
        return "low"
    elif value <= 3:
        return "medium"
    else:
        return "high"

def classify_alert_volume_60m(value: float) -> str:
    """
    Classifica volume recente de alertas em 60 min.
    """
    if pd.isna(value):
        return "unknown"
    elif value <= 2:
        return "low"
    elif value <= 5:
        return "medium"
    else:
        return "high"

def create_discretized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis categóricas interpretáveis para associação.
    """
    df["served_status"] = np.where(df["served_flag"], "served", "not_served")
    df["weekend_status"] = np.where(df["is_weekend"], "weekend", "weekday")

    df["response_time_class"] = df["response_seconds"].apply(classify_response_time)
    df["time_since_prev_alert_class"] = df["time_since_prev_alert_s"].apply(classify_time_since_prev_alert)
    df["alerts_last_15m_class"] = df["alerts_last_15m"].apply(classify_alert_volume_15m)
    df["alerts_last_60m_class"] = df["alerts_last_60m"].apply(classify_alert_volume_60m)

    return df


# Seleção de colunas e formatação variavel=valor para regras de associação

def select_association_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seleciona apenas colunas categóricas úteis para regras.
    """
    cols = [
        "hour_bin",
        "part_of_day",
        "day_of_week",
        "weekend_status",
        "time_since_prev_alert_class",
        "alerts_last_15m_class",
        "alerts_last_60m_class",
        "response_time_class",
        "served_status",
    ]
    return df[cols].copy()

def convert_to_item_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte cada coluna categórica para o formato:
    variavel=valor
    Isso facilita uso posterior em one-hot encoding e regras.
    """
    out = pd.DataFrame()

    for col in df.columns:
        out[col] = col + "=" + df[col].astype(str)

    return out

def discretization_categorical_pipeline(type_dataset: str):
    print("🔹 Carregando base limpa...")
    if type_dataset == "observational":
        df = load_data(OBSERVATIONAL_FEATURE_ENGINEERING_DATASET_PATH)
    elif type_dataset == "synthetic":
        df = load_data(SYNTETIC_FEATURE_ENGINEERING_DATASET_PATH)

    print("🔹 Convertendo datas...")
    df = parse_datetime(df)

    print("🔹 Criando faixas horárias...")
    df = create_hour_bin(df)

    print("🔹 Criando variáveis discretizadas...")
    df = create_discretized_features(df)

    print("🔹 Selecionando colunas para associação...")
    df_assoc = select_association_columns(df)

    print("🔹 Convertendo para formato item=valor...")
    df_assoc = convert_to_item_format(df_assoc)

    print("🔹 Salvando base final...")
    if type_dataset == "observational":
        df_assoc.to_csv(OBSERVATIONAL_DISCRETIZATION_CATEGORICAL_DATASET_PATH, sep=",", index=False)
    elif type_dataset == "synthetic":
        df_assoc.to_csv(SYNTETIC_DISCRETIZATION_CATEGORICAL_DATASET_PATH, sep=",", index=False)

    print("✅ Dataset discretisado para regras de associação gerado com sucesso!")
    print(df_assoc.head())
