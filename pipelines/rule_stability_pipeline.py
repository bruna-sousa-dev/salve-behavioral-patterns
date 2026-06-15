from pathlib import Path
import pandas as pd

from config import (
    APRIORI_OBSERVATIONAL_OUTPUT_PATH,
    FP_GROWTH_OBSERVATIONAL_OUTPUT_PATH,
    ECLAT_OBSERVATIONAL_OUTPUT_PATH,
    APRIORI_SYNTHETIC_OUTPUT_PATH,
    FP_GROWTH_SYNTHETIC_OUTPUT_PATH,
    ECLAT_SYNTHETIC_OUTPUT_PATH,
    RULE_OVERLAP_SUMMARY_ALL_ALGORITHMS_PATH,
    RULE_OVERLAP_SUMMARY_GENERIC_PATH,
    COMMON_RULES_OUTPUT_PATH,
)

def normalize_rule_key(row: pd.Series) -> str:
    """
    Create a normalized rule key based on antecedent and consequent.

    This avoids false differences caused by item ordering inside the antecedent.
    """

    antecedent_items = [
        item.strip()
        for item in str(row["antecedent"]).split(",")
        if item.strip()
    ]

    antecedent_items = sorted(antecedent_items)

    consequent = str(row["consequent"]).strip()

    return " AND ".join(antecedent_items) + " -> " + consequent


def load_rules_with_key(path: str) -> pd.DataFrame:
    """Load rules CSV and create normalized rule key."""

    df = pd.read_csv(path)

    required_columns = {
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}: {missing_columns}"
        )

    df["rule_key"] = df.apply(normalize_rule_key, axis=1)

    return df


def compute_rule_overlap(
    observational_rules_path: str,
    synthetic_rules_path: str,
    algorithm_name: str,
    output_dir: str = "outputARM",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rule stability/overlap between observational and synthetic datasets.

    Generates two CSV files:
    1. Summary metrics
    2. Common rules table
    """

    obs_df = load_rules_with_key(observational_rules_path)
    syn_df = load_rules_with_key(synthetic_rules_path)

    obs_rules = set(obs_df["rule_key"])
    syn_rules = set(syn_df["rule_key"])

    common_rules = obs_rules.intersection(syn_rules)

    obs_only_rules = obs_rules - syn_rules
    syn_only_rules = syn_rules - obs_rules

    n_obs = len(obs_rules)
    n_syn = len(syn_rules)
    n_common = len(common_rules)

    overlap_obs = n_common / n_obs if n_obs else 0
    overlap_syn = n_common / n_syn if n_syn else 0

    jaccard_similarity = (
        n_common / len(obs_rules.union(syn_rules))
        if obs_rules.union(syn_rules)
        else 0
    )

    summary_df = pd.DataFrame(
        [
            {
                "algorithm": algorithm_name,
                "observational_rules": n_obs,
                "synthetic_rules": n_syn,
                "common_rules": n_common,
                "observational_only_rules": len(obs_only_rules),
                "synthetic_only_rules": len(syn_only_rules),
                "overlap_relative_to_observational": overlap_obs,
                "overlap_relative_to_synthetic": overlap_syn,
                "jaccard_similarity": jaccard_similarity,
            }
        ]
    )

    common_rules_df = (
        obs_df[obs_df["rule_key"].isin(common_rules)]
        .merge(
            syn_df[["rule_key", "support", "confidence", "lift"]],
            on="rule_key",
            suffixes=("_observational", "_synthetic"),
        )
        .copy()
    )

    common_rules_df = common_rules_df[
        [
            "rule_key",
            "support_observational",
            "confidence_observational",
            "lift_observational",
            "support_synthetic",
            "confidence_synthetic",
            "lift_synthetic",
        ]
    ]

    common_rules_df = common_rules_df.sort_values(
        by=["lift_observational", "lift_synthetic"],
        ascending=[False, False],
    ).reset_index(drop=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_output = RULE_OVERLAP_SUMMARY_GENERIC_PATH.format(algorithm_name.lower())
    common_rules_output = COMMON_RULES_OUTPUT_PATH.format(algorithm_name.lower())

    summary_df.to_csv(summary_output, index=False, encoding="utf-8-sig")
    common_rules_df.to_csv(common_rules_output, index=False, encoding="utf-8-sig")

    print(f"[RULE_STABILITY] Summary exported to: {summary_output}")
    print(f"[RULE_STABILITY] Common rules exported to: {common_rules_output}")

    return summary_df, common_rules_df


def compute_all_rule_overlaps() -> None:
    """Compute overlap for Apriori, FP-Growth, and ECLAT."""

    all_summaries = []

    experiments = [
        {
            "algorithm_name": "Apriori",
            "observational_rules_path": APRIORI_OBSERVATIONAL_OUTPUT_PATH,
            "synthetic_rules_path": APRIORI_SYNTHETIC_OUTPUT_PATH,
        },
        {
            "algorithm_name": "FPGrowth",
            "observational_rules_path": FP_GROWTH_OBSERVATIONAL_OUTPUT_PATH,
            "synthetic_rules_path": FP_GROWTH_SYNTHETIC_OUTPUT_PATH,
        },
        {
            "algorithm_name": "ECLAT",
            "observational_rules_path": ECLAT_OBSERVATIONAL_OUTPUT_PATH,
            "synthetic_rules_path": ECLAT_SYNTHETIC_OUTPUT_PATH,
        },
    ]

    for experiment in experiments:
        summary_df, _ = compute_rule_overlap(
            observational_rules_path=experiment["observational_rules_path"],
            synthetic_rules_path=experiment["synthetic_rules_path"],
            algorithm_name=experiment["algorithm_name"],
            output_dir="outputARM",
        )

        all_summaries.append(summary_df)

    final_summary_df = pd.concat(all_summaries, ignore_index=True)

    final_output = RULE_OVERLAP_SUMMARY_ALL_ALGORITHMS_PATH

    final_summary_df.to_csv(final_output, index=False, encoding="utf-8-sig")

    print(f"[RULE_STABILITY] Final summary exported to: {final_output}")
