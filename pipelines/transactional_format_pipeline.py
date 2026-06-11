from pathlib import Path

import pandas as pd

from config import (
    OBSERVATIONAL_DISCRETIZATION_CATEGORICAL_DATASET_PATH,
    SYNTETIC_DISCRETIZATION_CATEGORICAL_DATASET_PATH,
    OBSERVATIONAL_TRANSACTIONAL_DATASET_PATH,
    SYNTETIC_TRANSACTIONAL_DATASET_PATH
)

# **************************************************************************************
# Pipeline para conversão da base categórica discretizada em formato transacional


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Carrega a base categórica discretizada com separador ','.
    """
    df = pd.read_csv(path, sep=",")
    return df


# Conversão para formato transacional


def select_transactional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seleciona apenas as colunas categóricas utilizadas na representação transacional.
    """

    cols = [
        "hour_bin",
        # "part_of_day",
        "day_of_week",
        "weekend_status",
        "time_since_prev_alert_class",
        "alerts_last_15m_class",
        "alerts_last_60m_class",
        "response_time_class",
        # "served_status",
    ]

    missing_columns = [col for col in cols if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"As seguintes colunas obrigatórias não foram encontradas: {missing_columns}"
        )

    return df[cols].dropna().copy()


def convert_to_transactional_format(df: pd.DataFrame) -> list[list[str]]:
    """
    Converte a base categórica discretizada para formato transacional.

    Cada linha do dataset passa a representar uma transação.

    Como a base de entrada já está no formato variavel=valor,
    não é realizada nova discretização nesta etapa.

    Exemplo de transação:
    [
        "hour_bin=09_10",
        "part_of_day=morning",
        "day_of_week=2",
        "weekend_status=weekday",
        "time_since_prev_alert_class=short_le_5m",
        "alerts_last_15m_class=low",
        "alerts_last_60m_class=low",
        "response_time_class=normal_61_180s",
        "served_status=served"
    ]
    """

    records = []

    for _, row in df.iterrows():
        transaction = [str(value) for value in row.values]
        records.append(transaction)

    return records


# Exportação da base transacional


def export_transactional_dataset(
    records: list[list[str]],
    output_path: str | Path,
) -> None:
    """
    Exporta a base transacional para CSV.

    Cada transação é salva como uma linha.
    Cada item da transação é salvo em uma coluna:
    item_1, item_2, item_3, ...
    """

    if not records:
        raise ValueError("Nenhuma transação disponível para exportação.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_transaction_size = max(len(transaction) for transaction in records)

    columns = [
        f"item_{index + 1}"
        for index in range(max_transaction_size)
    ]

    transactional_df = pd.DataFrame(records, columns=columns)

    transactional_df.to_csv(output_path, sep=",", index=False, encoding="utf-8-sig")


def transactional_format_pipeline(type_dataset: str) -> list[list[str]]:
    """
    Executa apenas a etapa de conversão para formato transacional.
    """

    print("[TRANSACTIONAL_FORMAT] Carregando base categórica discretizada...")
    if type_dataset == "observational":
        df = load_data(OBSERVATIONAL_DISCRETIZATION_CATEGORICAL_DATASET_PATH)
    elif type_dataset == "synthetic":
        df = load_data(SYNTETIC_DISCRETIZATION_CATEGORICAL_DATASET_PATH)

    print("[TRANSACTIONAL_FORMAT] Selecionando colunas transacionais...")
    df_transactional = select_transactional_columns(df)

    print("[TRANSACTIONAL_FORMAT] Convertendo para formato transacional...")
    records = convert_to_transactional_format(df_transactional)

    print(f"[TRANSACTIONAL_FORMAT] Número de transações geradas: {len(records)}")

    if records:
        print("[TRANSACTIONAL_FORMAT] Exemplo de transação:")
        print(records[0])

    print("[TRANSACTIONAL_FORMAT] Salvando base transacional...")
    if type_dataset == "observational":
        export_transactional_dataset(records, OBSERVATIONAL_TRANSACTIONAL_DATASET_PATH)
    elif type_dataset == "synthetic":
        export_transactional_dataset(records, SYNTETIC_TRANSACTIONAL_DATASET_PATH)

    print("[TRANSACTIONAL_FORMAT] Dataset transacional gerado com sucesso!")

    return records