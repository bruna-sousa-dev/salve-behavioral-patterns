import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Figure 9 - Execution time of ARM algorithms
# ============================================================

# Ajuste o caminho se o arquivo estiver em outra pasta
RESULTS_ALGORITHM_COMPARISON_TABLE_PATH = Path("outputARM/results_algorithm_comparison_table.csv")
EXECUTION_TIME_OF_ALGORITHMS_PATH = Path("images/execution_time_of_algorithms.png")


def load_algorithm_comparison(path: Path) -> pd.DataFrame:
    """
    Carrega a tabela comparativa dos algoritmos.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required_columns = [
        "Dataset",
        "Algorithm",
        "Time (s)",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in file: {missing_columns}")

    return df

if __name__ == "__main__":
    # Load data
    df = load_algorithm_comparison(RESULTS_ALGORITHM_COMPARISON_TABLE_PATH)

    # Padronização dos nomes para exibição no gráfico
    df["Dataset"] = df["Dataset"].replace({
        "observational": "Observational",
        "synthetic": "Synthetic",
    })

    df["Algorithm"] = df["Algorithm"].replace({
        "ECLAT": "Eclat",
    })

    # Ordem desejada no gráfico
    algorithm_order = ["Apriori", "FP-Growth", "Eclat"]
    dataset_order = ["Observational", "Synthetic"]

    df["Algorithm"] = pd.Categorical(
        df["Algorithm"],
        categories=algorithm_order,
        ordered=True,
    )

    df["Dataset"] = pd.Categorical(
        df["Dataset"],
        categories=dataset_order,
        ordered=True,
    )

    df = df.sort_values(["Algorithm", "Dataset"])

    # Pivot para gráfico de barras agrupadas
    plot_df = df.pivot(
        index="Algorithm",
        columns="Dataset",
        values="Time (s)",
    )

    # Plot
    ax = plot_df.plot(
        kind="bar",
        figsize=(8, 5),
        width=0.75,
    )

    # ax.set_title(
    #     "Execution time of association rule mining algorithms",
    #     fontsize=13,
    # )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Execution time (s)")

    ax.legend(title="Dataset")
    ax.tick_params(axis="x", rotation=0)

    # Valores acima das barras
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9)

    plt.tight_layout()

    # Save figure
    plt.savefig(EXECUTION_TIME_OF_ALGORITHMS_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved as: {EXECUTION_TIME_OF_ALGORITHMS_PATH}")
