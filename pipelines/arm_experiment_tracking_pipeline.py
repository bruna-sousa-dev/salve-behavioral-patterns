import time
import platform
import sys
import pandas as pd
from typing import Any, Callable


def run_with_timer(
    algorithm_name: str,
    algorithm_function: Callable,
    records: list[list[str]],
):
    """Run ARM algorithm and measure execution time."""

    start_time = time.perf_counter()
    results = algorithm_function(records=records)
    end_time = time.perf_counter()

    execution_time_s = end_time - start_time

    print(f"[{algorithm_name}] Tempo de execução: {execution_time_s:.6f} s")

    return results, execution_time_s


def get_result_metadata(results: Any) -> dict:
    """Recover metadata from ARM results."""

    if isinstance(results, pd.DataFrame):
        return {
            "frequent_itemsets_count": results.attrs.get("frequent_itemsets_count"),
            "rules_generated_count": results.attrs.get("rules_generated_count"),
            "min_support": results.attrs.get("min_support"),
            "min_confidence": results.attrs.get("min_confidence"),
            "min_lift": results.attrs.get("min_lift"),
            "min_length": results.attrs.get("min_length"),
        }

    return {
        "frequent_itemsets_count": getattr(results, "frequent_itemsets_count", None),
        "rules_generated_count": getattr(results, "rules_generated_count", None),
        "min_support": getattr(results, "min_support", None),
        "min_confidence": getattr(results, "min_confidence", None),
        "min_lift": getattr(results, "min_lift", None),
        "min_length": getattr(results, "min_length", None),
    }


def summarize_rules_metrics(rules_df: pd.DataFrame) -> dict:
    """Generate descriptive statistics for final filtered rules."""

    if rules_df.empty:
        return {
            "rules_after_filter": 0,
            "mean_support": None,
            "median_support": None,
            "max_support": None,
            "mean_confidence": None,
            "median_confidence": None,
            "max_confidence": None,
            "mean_lift": None,
            "median_lift": None,
            "max_lift": None,
            "not_served_rules": 0,
            "fast_rules": 0,
            "normal_rules": 0,
            "slow_rules": 0,
        }

    return {
        "rules_after_filter": len(rules_df),
        "mean_support": rules_df["support"].mean(),
        "median_support": rules_df["support"].median(),
        "max_support": rules_df["support"].max(),
        "mean_confidence": rules_df["confidence"].mean(),
        "median_confidence": rules_df["confidence"].median(),
        "max_confidence": rules_df["confidence"].max(),
        "mean_lift": rules_df["lift"].mean(),
        "median_lift": rules_df["lift"].median(),
        "max_lift": rules_df["lift"].max(),
        "not_served_rules": (rules_df["consequent"] == "response_time_class=not_served").sum(),
        "fast_rules": (rules_df["consequent"] == "response_time_class=fast_le_60s").sum(),
        "normal_rules": (rules_df["consequent"] == "response_time_class=normal_61_180s").sum(),
        "slow_rules": (rules_df["consequent"] == "response_time_class=slow_gt_180s").sum(),
    }


def build_experiment_summary_row(
    dataset_type: str,
    algorithm_name: str,
    records: list[list[str]],
    raw_results: Any,
    filtered_rules_df: pd.DataFrame,
    execution_time_s: float,
    target_consequents: set[str],
    min_antecedent_size: int,
    max_antecedent_size: int,
) -> dict:
    """Build one traceability row for one ARM experiment."""

    metadata = get_result_metadata(raw_results)
    metrics = summarize_rules_metrics(filtered_rules_df)

    unique_items = sorted(set(item for transaction in records for item in transaction))

    row = {
        "dataset_type": dataset_type,
        "algorithm": algorithm_name,
        "n_transactions": len(records),
        "n_unique_items": len(unique_items),
        "min_support": metadata["min_support"],
        "min_confidence": metadata["min_confidence"],
        "min_lift": metadata["min_lift"],
        "min_length": metadata["min_length"],
        "target_consequents": " | ".join(sorted(target_consequents)),
        "min_antecedent_size": min_antecedent_size,
        "max_antecedent_size": max_antecedent_size,
        "frequent_itemsets_count": metadata["frequent_itemsets_count"],
        "rules_generated_before_export_filter": metadata["rules_generated_count"],
        "execution_time_s": execution_time_s,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    row.update(metrics)

    return row


def export_experiment_summary(
    summary_rows: list[dict],
    output_path: str,
) -> None:
    """Export experiment summary to CSV."""

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[EXPERIMENT_SUMMARY] Exportado para: {output_path}")
