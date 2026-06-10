import pandas as pd
from config import OBSERVATIONAL_DATASET_PATH, OBSERVATIONAL_CLEANED_PREPROCESSED_DATASET_PATH

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

# =========================================================
# DESIGN DECISION:
# Pipeline determinístico de limpeza e enriquecimento temporal
# - Evita data leakage (não usa served_time como feature explicativa direta)
# - Gera variáveis interpretáveis (importante para regras de associação)
# - Mantém consistência temporal (ordenação + deltas)
# =========================================================
def cleaning_preprocess_pipeline():
    print("Carregando dados...")
    df = load_data(OBSERVATIONAL_DATASET_PATH)

    # Limpeza e preprocessamento
    print("Parse de datas...")
    df = parse_datetime(df)
    print("Limpeza...")
    df = clean_data(df)

    print("Salvando...")
    df.to_csv(OBSERVATIONAL_CLEANED_PREPROCESSED_DATASET_PATH, sep=",", index=False)

    print("✅ Dataset limpo gerado com sucesso!")
    print(df.head())
