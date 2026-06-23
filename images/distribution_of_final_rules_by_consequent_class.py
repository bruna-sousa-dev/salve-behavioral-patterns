import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Figure 8 - Distribution of final rules by consequent class
# ============================================================

ARM_EXPERIMENT_SUMMARY_OBSERVATIONAL_PATH = Path("outputARM/arm_experiment_summary_observational.csv")
ARM_EXPERIMENT_SUMMARY_SYNTHETIC_PATH = Path("outputARM/arm_experiment_summary_synthetic.csv")
FIGURE_RULES_BY_CONSEQUENT_CLASS_PATH = Path("images/distribution_of_final_rules_by_consequent_class.png")

def load_rule_summary(path: Path) -> pd.DataFrame:
    """
    Carrega o arquivo de resumo dos experimentos ARM.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def extract_consequent_counts(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """
    Extrai a quantidade de regras finais por classe consequente.

    Os arquivos possuem uma linha por algoritmo, mas as contagens por consequente
    são iguais para Apriori, FP-Growth e Eclat dentro do mesmo dataset.
    Por isso, usa-se a primeira linha como representação do dataset.
    """
    required_columns = [
        "not_served_rules",
        "fast_rules",
        "normal_rules",
        "slow_rules",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset summary: {missing_columns}")

    row = df.iloc[0]

    data = {
        "Dataset": dataset_label,
        "Fast response": int(row["fast_rules"]),
        "Normal response": int(row["normal_rules"]),
        "Slow response": int(row["slow_rules"]),
        "Not served": int(row["not_served_rules"]),
    }

    return pd.DataFrame([data])


if __name__ == "__main__":
    # Load datasets
    observational_df = load_rule_summary(ARM_EXPERIMENT_SUMMARY_OBSERVATIONAL_PATH)
    synthetic_df = load_rule_summary(ARM_EXPERIMENT_SUMMARY_SYNTHETIC_PATH)

    # Extract counts
    observational_counts = extract_consequent_counts(
        observational_df,
        dataset_label="Observational",
    )

    synthetic_counts = extract_consequent_counts(
        synthetic_df,
        dataset_label="Synthetic",
    )

    # Combine data
    plot_df = pd.concat(
        [observational_counts, synthetic_counts],
        ignore_index=True,
    )

    # Convert to long format
    plot_long_df = plot_df.melt(
        id_vars="Dataset",
        var_name="Consequent class",
        value_name="Number of rules",
    )

    # Pivot for grouped bar chart
    pivot_df = plot_long_df.pivot(
        index="Consequent class",
        columns="Dataset",
        values="Number of rules",
    )

    # Ordem das classes no gráfico
    class_order = [
        "Fast response",
        "Normal response",
        "Slow response",
        "Not served",
    ]

    pivot_df = pivot_df.loc[class_order]

    # Plot
    ax = pivot_df.plot(
        kind="bar",
        figsize=(9, 5),
        width=0.75,
    )

    # ax.set_title(
    #     "Distribution of final rules by consequent class",
    #     fontsize=13,
    # )

    ax.set_xlabel("Consequent class")
    ax.set_ylabel("Number of rules")

    ax.legend(title="Dataset")
    ax.tick_params(axis="x", rotation=0)

    # Add values above bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

    plt.tight_layout()

    # Save figure
    plt.savefig(FIGURE_RULES_BY_CONSEQUENT_CLASS_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved as: {FIGURE_RULES_BY_CONSEQUENT_CLASS_PATH}")
