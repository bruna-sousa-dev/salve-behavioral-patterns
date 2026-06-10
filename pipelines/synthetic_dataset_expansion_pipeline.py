"""
Controlled synthetic expansion for the SALVE observational dataset.

Final output schema:
- baia_id
- alert_type
- alert_time
- served_time
- served_flag
- bat_dist
- bat_cont
- data_origin
- reference_real_index

Notes:
- response_seconds is used only as an internal helper variable for generation,
  consistency validation, and validation reporting.
- response_seconds is not saved in the expanded dataset or in the synthetic-only dataset.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    OBSERVATIONAL_CLEANED_PREPROCESSED_DATASET_PATH,
    SYNTHETIC_EXPANDED_DATASET_PATH,
    SYNTHETIC_DATASET_ONLY_PATH,
    SYNTHETIC_EXPANSION_VALIDATION_REPORT,
)

SEED = 42
FINAL_SIZE = 1000

OUTPUT_COLUMNS = [
    "baia_id",
    "alert_type",
    "alert_time",
    "served_time",
    "served_flag",
    "bat_dist",
    "bat_cont",
    "data_origin",
    "reference_real_index",
]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "baia_id",
        "alert_type",
        "alert_time",
        "served_time",
        "served_flag",
        "bat_dist",
        "bat_cont",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["alert_time"] = pd.to_datetime(df["alert_time"], errors="coerce")
    df["served_time"] = pd.to_datetime(df["served_time"], errors="coerce")

    if df["served_flag"].dtype != bool:
        df["served_flag"] = (
            df["served_flag"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    if df["alert_time"].isna().any():
        raise ValueError("There are invalid alert_time values.")

    if df["served_flag"].isna().any():
        raise ValueError("There are invalid served_flag values.")

    # Internal helper only. It is not saved in the final datasets.
    df["response_seconds"] = (df["served_time"] - df["alert_time"]).dt.total_seconds()
    return df


def jitter_time_from_reference(
    ref_alert_time: pd.Timestamp,
    target_date,
    rng: np.random.Generator,
) -> pd.Timestamp:
    """Preserve the reference time pattern with a small bounded jitter."""
    base = pd.Timestamp.combine(pd.Timestamp(target_date).date(), ref_alert_time.time())

    jitter_seconds = int(rng.integers(-300, 301))
    candidate = base + pd.Timedelta(seconds=jitter_seconds)

    lower = pd.Timestamp.combine(pd.Timestamp(target_date).date(), pd.Timestamp("07:00:00").time())
    upper = pd.Timestamp.combine(pd.Timestamp(target_date).date(), pd.Timestamp("15:59:59").time())

    if candidate < lower:
        candidate = lower + pd.Timedelta(seconds=int(rng.integers(0, 600)))
    if candidate > upper:
        candidate = upper - pd.Timedelta(seconds=int(rng.integers(0, 600)))

    return candidate


def resample_response_time(real_response_times: np.ndarray, rng: np.random.Generator) -> int:
    """Resample empirical response time with a small bounded perturbation."""
    min_response = int(np.nanmin(real_response_times))
    max_response = int(np.nanmax(real_response_times))

    base = float(rng.choice(real_response_times))
    sd = max(base * 0.10, 5.0)
    value = rng.normal(loc=base, scale=sd)

    return int(round(np.clip(value, min_response, max_response)))


def build_day_pool(df: pd.DataFrame, n_synthetic: int, rng: np.random.Generator) -> np.ndarray:
    """Create synthetic dates preserving the observed daily alert volume."""
    daily_counts = df.groupby(df["alert_time"].dt.date).size().to_numpy()
    mean_daily_count = daily_counts.mean()
    n_synthetic_days = ceil(n_synthetic / mean_daily_count)

    start_synthetic_date = df["alert_time"].dt.date.max() + pd.Timedelta(days=1)
    synthetic_dates = pd.date_range(
        start=start_synthetic_date,
        periods=n_synthetic_days,
        freq="D",
    ).date

    day_pool = []
    while len(day_pool) < n_synthetic:
        for date in synthetic_dates:
            count = int(rng.choice(daily_counts))
            day_pool.extend([date] * count)
            if len(day_pool) >= n_synthetic:
                break

    day_pool = np.array(day_pool[:n_synthetic], dtype=object)
    rng.shuffle(day_pool)
    return day_pool


def generate_synthetic_records(
    df: pd.DataFrame,
    final_size: int = FINAL_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    n_real = len(df)
    n_synthetic = final_size - n_real

    if n_synthetic <= 0:
        raise ValueError("final_size must be greater than the number of real records.")

    df = df.copy().reset_index(drop=False).rename(columns={"index": "original_real_index"})

    attended_ref = df[df["served_flag"]].copy().reset_index(drop=True)
    unattended_ref = df[~df["served_flag"]].copy().reset_index(drop=True)

    if attended_ref.empty or unattended_ref.empty:
        raise ValueError("The dataset must contain attended and unattended records.")

    n_attended_real = int(df["served_flag"].sum())
    target_attended_final = round(final_size * (n_attended_real / n_real))
    target_unattended_final = final_size - target_attended_final

    n_attended_synthetic = target_attended_final - n_attended_real
    n_unattended_synthetic = target_unattended_final - int((~df["served_flag"]).sum())

    day_pool = build_day_pool(df, n_synthetic, rng)

    status_pool = np.array(
        [True] * n_attended_synthetic + [False] * n_unattended_synthetic,
        dtype=bool,
    )
    rng.shuffle(status_pool)

    real_response_times = attended_ref["response_seconds"].dropna().to_numpy()
    synthetic_rows = []

    for target_date, served_flag in zip(day_pool, status_pool):
        if served_flag:
            ref = attended_ref.iloc[int(rng.integers(0, len(attended_ref)))]
        else:
            ref = unattended_ref.iloc[int(rng.integers(0, len(unattended_ref)))]

        new_alert_time = jitter_time_from_reference(ref["alert_time"], target_date, rng)

        if served_flag:
            response_seconds = resample_response_time(real_response_times, rng)
            new_served_time = new_alert_time + pd.Timedelta(seconds=response_seconds)
        else:
            response_seconds = np.nan
            new_served_time = pd.NaT

        synthetic_rows.append(
            {
                "baia_id": ref["baia_id"],
                "alert_type": ref["alert_type"],
                "alert_time": new_alert_time,
                "served_time": new_served_time,
                "served_flag": bool(served_flag),
                "bat_dist": ref["bat_dist"],
                "bat_cont": ref["bat_cont"],
                "data_origin": "synthetic",
                "reference_real_index": int(ref["original_real_index"]),
                "response_seconds": response_seconds,  # internal helper only
            }
        )

    synthetic = pd.DataFrame(synthetic_rows).sort_values("alert_time").reset_index(drop=True)

    duplicated_mask = synthetic["alert_time"].duplicated(keep=False)
    if duplicated_mask.any():
        for _, idxs in synthetic[duplicated_mask].groupby("alert_time").groups.items():
            for offset, idx in enumerate(list(idxs)):
                synthetic.loc[idx, "alert_time"] = synthetic.loc[idx, "alert_time"] + pd.Timedelta(seconds=offset)
                if synthetic.loc[idx, "served_flag"]:
                    synthetic.loc[idx, "served_time"] = synthetic.loc[idx, "alert_time"] + pd.Timedelta(
                        seconds=int(synthetic.loc[idx, "response_seconds"])
                    )

    real_out = df.copy()
    real_out["data_origin"] = "real"
    real_out["reference_real_index"] = np.nan

    expanded = pd.concat([real_out, synthetic], ignore_index=True)
    expanded = expanded.sort_values("alert_time").reset_index(drop=True)
    expanded["response_seconds"] = (expanded["served_time"] - expanded["alert_time"]).dt.total_seconds()

    validate_logical_consistency(expanded, final_size)

    return expanded, synthetic


def validate_logical_consistency(df: pd.DataFrame, expected_size: int) -> None:
    if len(df) != expected_size:
        raise ValueError(f"Final dataset has {len(df)} records, expected {expected_size}.")

    attended_without_served_time = df[df["served_flag"] & df["served_time"].isna()]
    unattended_with_served_time = df[(~df["served_flag"]) & df["served_time"].notna()]
    nonpositive_response = df[df["served_flag"] & (df["response_seconds"] <= 0)]

    if not attended_without_served_time.empty:
        raise ValueError("There are attended records without served_time.")
    if not unattended_with_served_time.empty:
        raise ValueError("There are unattended records with served_time.")
    if not nonpositive_response.empty:
        raise ValueError("There are attended records with non-positive response time.")


def summary_metrics(data: pd.DataFrame, label: str) -> dict:
    temp = data.copy()
    temp["response_seconds"] = (temp["served_time"] - temp["alert_time"]).dt.total_seconds()
    response = temp.loc[temp["served_flag"], "response_seconds"].dropna()
    daily_counts = temp.groupby(temp["alert_time"].dt.date).size()

    return {
        "dataset": label,
        "records": len(temp),
        "attended": int(temp["served_flag"].sum()),
        "unattended": int((~temp["served_flag"]).sum()),
        "attendance_rate": float(temp["served_flag"].mean()),
        "alert_start": temp["alert_time"].min(),
        "alert_end": temp["alert_time"].max(),
        "response_count": int(response.count()),
        "response_mean_s": float(response.mean()),
        "response_median_s": float(response.median()),
        "response_std_s": float(response.std()),
        "response_min_s": float(response.min()),
        "response_q1_s": float(response.quantile(0.25)),
        "response_q3_s": float(response.quantile(0.75)),
        "response_max_s": float(response.max()),
        "min_hour": int(temp["alert_time"].dt.hour.min()),
        "max_hour": int(temp["alert_time"].dt.hour.max()),
        "daily_count_mean": float(daily_counts.mean()),
        "daily_count_min": int(daily_counts.min()),
        "daily_count_max": int(daily_counts.max()),
    }


def format_output_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output = output[OUTPUT_COLUMNS]

    for col in ["alert_time", "served_time"]:
        output[col] = pd.to_datetime(output[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    output["served_time"] = output["served_time"].fillna("")
    return output


def save_csv(df: pd.DataFrame, path: Path) -> None:
    output = format_output_for_csv(df)
    output.to_csv(path, index=False)


def synthetic_dataset_expansion_pipeline() -> None:
    df = load_dataset(OBSERVATIONAL_CLEANED_PREPROCESSED_DATASET_PATH)

    expanded, synthetic = generate_synthetic_records(
        df=df,
        final_size=FINAL_SIZE,
        seed=SEED,
    )

    report = pd.DataFrame(
        [
            summary_metrics(df.assign(data_origin="real"), "real"),
            summary_metrics(synthetic, "synthetic_only"),
            summary_metrics(expanded, "expanded_real_plus_synthetic"),
        ]
    )

    save_csv(expanded, SYNTHETIC_EXPANDED_DATASET_PATH)
    save_csv(synthetic, SYNTHETIC_DATASET_ONLY_PATH)
    report.to_csv(SYNTHETIC_EXPANSION_VALIDATION_REPORT, index=False)

    print("Synthetic expansion completed.")
    print(f"Expanded dataset: {SYNTHETIC_EXPANDED_DATASET_PATH}")
    print(f"Synthetic-only dataset: {SYNTHETIC_DATASET_ONLY_PATH}")
    print(f"Validation report: {SYNTHETIC_EXPANSION_VALIDATION_REPORT}")
    print(report)


if __name__ == "__main__":
    synthetic_dataset_expansion_pipeline()
