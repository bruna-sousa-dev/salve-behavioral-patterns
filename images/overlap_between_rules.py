import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Figure 10 - Overlap between observational and synthetic rules
# ============================================================

# Ajuste o caminho se o arquivo estiver em outra pasta
RULE_OVERLAP_SUMMARY_ALL_ALGORITHMS_PATH = Path("outputARM/rule_overlap_summary_all_algorithms.csv")
OVERLAP_BETWEEN_RULES = Path("images/overlap_between_rules.png")


def load_rule_overlap_summary(path: Path) -> pd.DataFrame:
    """
    Carrega o resumo de sobreposição entre regras observacionais e sintéticas.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required_columns = [
        "algorithm",
        "common_rules",
        "observational_only_rules",
        "synthetic_only_rules",
        "overlap_relative_to_observational",
        "overlap_relative_to_synthetic",
        "jaccard_similarity",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in file: {missing_columns}")

    return df

if __name__ == "__main__":
    # Load data
    df = load_rule_overlap_summary(RULE_OVERLAP_SUMMARY_ALL_ALGORITHMS_PATH)

    # Padronização dos nomes dos algoritmos para exibição
    df["algorithm"] = df["algorithm"].replace({
        "FPGrowth": "FP-Growth",
        "ECLAT": "Eclat",
    })

    # Ordem desejada no gráfico
    algorithm_order = ["Apriori", "FP-Growth", "Eclat"]

    df["algorithm"] = pd.Categorical(
        df["algorithm"],
        categories=algorithm_order,
        ordered=True,
    )

    df = df.sort_values("algorithm")

    # Seleção das colunas para o gráfico
    plot_df = df.set_index("algorithm")[
        [
            "common_rules",
            "observational_only_rules",
            "synthetic_only_rules",
        ]
    ]

    # Renomear colunas para legenda
    plot_df = plot_df.rename(
        columns={
            "common_rules": "Common rules",
            "observational_only_rules": "Observational only",
            "synthetic_only_rules": "Synthetic only",
        }
    )

    # Plot
    ax = plot_df.plot(
        kind="bar",
        figsize=(9, 5),
        width=0.75,
    )

    # ax.set_title(
    #     "Overlap between observational and synthetic association rules",
    #     fontsize=13,
    # )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Number of rules")

    ax.legend(title="Rule category")
    ax.tick_params(axis="x", rotation=0)

    # Adicionar valores acima das barras
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

    plt.tight_layout()

    # Save figure
    plt.savefig(OVERLAP_BETWEEN_RULES, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved as: {OVERLAP_BETWEEN_RULES}")
