from pathlib import Path
import pandas as pd

from config import (
    ARM_EXPERIMENT_SUMMARY_OBSERVATIONAL_PATH,
    ARM_EXPERIMENT_SUMMARY_SYNTHETIC_PATH,
    APRIORI_OBSERVATIONAL_OUTPUT_PATH,
    FP_GROWTH_OBSERVATIONAL_OUTPUT_PATH,
    ECLAT_OBSERVATIONAL_OUTPUT_PATH,
    APRIORI_SYNTHETIC_OUTPUT_PATH,
    FP_GROWTH_SYNTHETIC_OUTPUT_PATH,
    ECLAT_SYNTHETIC_OUTPUT_PATH,
    RESULTS_ALGORITHM_COMPARISON_TABLE_PATH,
    RESULTS_TOP10_RULES_OBSERVATIONAL_PATH,
    RESULTS_TOP10_RULES_SYNTHETIC_PATH,
)

def load_rules(path: str) -> pd.DataFrame:
    """Load ARM rules CSV."""

    df = pd.read_csv(path)

    required_cols = {
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift",
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing columns in {path}: {missing_cols}"
        )

    return df


def load_experiment_summary(path: str) -> pd.DataFrame:
    """Load ARM experiment summary CSV."""

    df = pd.read_csv(path)

    required_cols = {
        "dataset_type",
        "algorithm",
        "rules_after_filter",
        "mean_support",
        "mean_confidence",
        "mean_lift",
        "execution_time_s",
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing columns in {path}: {missing_cols}"
        )

    return df


def generate_algorithm_comparison_table(
    observational_summary_path: str,
    synthetic_summary_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Generate table:
    Dataset | Algorithm | Rules Final | Mean Support | Mean Confidence | Mean Lift | Time (s)
    """

    observational_df = load_experiment_summary(observational_summary_path)
    synthetic_df = load_experiment_summary(synthetic_summary_path)

    df = pd.concat(
        [observational_df, synthetic_df],
        ignore_index=True,
    )

    table = df[
        [
            "dataset_type",
            "algorithm",
            "rules_after_filter",
            "mean_support",
            "mean_confidence",
            "mean_lift",
            "execution_time_s",
        ]
    ].copy()

    table = table.rename(
        columns={
            "dataset_type": "Dataset",
            "algorithm": "Algorithm",
            "rules_after_filter": "Rules Final",
            "mean_support": "Mean Support",
            "mean_confidence": "Mean Confidence",
            "mean_lift": "Mean Lift",
            "execution_time_s": "Time (s)",
        }
    )

    table["Mean Support"] = table["Mean Support"].round(4)
    table["Mean Confidence"] = table["Mean Confidence"].round(4)
    table["Mean Lift"] = table["Mean Lift"].round(4)
    table["Time (s)"] = table["Time (s)"].round(6)

    table = table.sort_values(
        by=["Dataset", "Algorithm"],
        ascending=[True, True],
    ).reset_index(drop=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"[RESULTS_TABLE] Algorithm comparison table exported to: {output_path}")

    return table


def generate_top_rules_table(
    rules_paths: dict[str, str],
    dataset_type: str,
    output_path: str,
    top_n: int = 10,
    remove_trivial_rules: bool = True,
) -> pd.DataFrame:
    """
    Generate table:
    Rank | Rule | Support | Confidence | Lift

    rules_paths example:
    {
        "Apriori": "outputARM/apriori_observational_output.csv",
        "FP-Growth": "outputARM/fpgrowth_observational_output.csv",
        "ECLAT": "outputARM/eclat_observational_output.csv",
    }
    """

    all_rules = []

    for algorithm, path in rules_paths.items():
        df = load_rules(path)

        df["Algorithm"] = algorithm
        df["Dataset"] = dataset_type

        all_rules.append(df)

    rules_df = pd.concat(all_rules, ignore_index=True)

    if remove_trivial_rules:
        rules_df = rules_df[
            ~(
                (rules_df["antecedent"].str.contains("served_status=not_served", na=False))
                & (rules_df["consequent"] == "response_time_class=not_served")
            )
        ].copy()

    rules_df["Rule"] = (
        rules_df["antecedent"].astype(str)
        + " -> "
        + rules_df["consequent"].astype(str)
    )

    # Remove duplicates because Apriori, FP-Growth and ECLAT may generate the same rules.
    rules_df = rules_df.drop_duplicates(
        subset=["Rule", "support", "confidence", "lift"]
    )

    rules_df = rules_df.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).head(top_n)

    table = rules_df[
        [
            "Rule",
            "support",
            "confidence",
            "lift",
        ]
    ].copy()

    table.insert(0, "Rank", range(1, len(table) + 1))

    table = table.rename(
        columns={
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
        }
    )

    table["Support"] = table["Support"].round(4)
    table["Confidence"] = table["Confidence"].round(4)
    table["Lift"] = table["Lift"].round(4)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"[RESULTS_TABLE] Top {top_n} rules table exported to: {output_path}")

    return table


def generate_all_results_tables() -> None:
    """Generate all CSV tables used in Results and Discussion."""

    generate_algorithm_comparison_table(
        observational_summary_path=ARM_EXPERIMENT_SUMMARY_OBSERVATIONAL_PATH,
        synthetic_summary_path=ARM_EXPERIMENT_SUMMARY_SYNTHETIC_PATH,
        output_path=RESULTS_ALGORITHM_COMPARISON_TABLE_PATH,
    )

    generate_top_rules_table(
        rules_paths={
            "Apriori": APRIORI_OBSERVATIONAL_OUTPUT_PATH,
            "FP-Growth": FP_GROWTH_OBSERVATIONAL_OUTPUT_PATH,
            "ECLAT": ECLAT_OBSERVATIONAL_OUTPUT_PATH,
        },
        dataset_type="observational",
        output_path=RESULTS_TOP10_RULES_OBSERVATIONAL_PATH,
        top_n=10,
    )

    generate_top_rules_table(
        rules_paths={
            "Apriori": APRIORI_SYNTHETIC_OUTPUT_PATH,
            "FP-Growth": FP_GROWTH_SYNTHETIC_OUTPUT_PATH,
            "ECLAT": ECLAT_SYNTHETIC_OUTPUT_PATH,
        },
        dataset_type="synthetic",
        output_path=RESULTS_TOP10_RULES_SYNTHETIC_PATH,
        top_n=10,
    )
