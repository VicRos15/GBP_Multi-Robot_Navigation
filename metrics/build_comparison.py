#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import pandas as pd


METRICS_TO_ANALYZE = [
    "makespan_sim",
    "real_time_total",
    "total_distance",
    "mean_distance",
    "max_distance",
    "mean_obstacle_clearance",
    "mean_agent_clearance",
    "mean_safety_clearance",
    "mean_speed",
    "max_speed",
    "mean_smoothness_ldj",
    "mean_path_curvature",
]


def load_metrics_files(paths: list[str]) -> pd.DataFrame:
    dfs = []

    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path
        dfs.append(df)

    if not dfs:
        raise ValueError("No se ha cargado ningún CSV.")

    return pd.concat(dfs, ignore_index=True)


def normalize_success_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "success" not in df.columns:
        df["success"] = False
        return df

    df["success"] = (
        df["success"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["env_type", "algorithm"]

    for (env_type, algorithm), group in df.groupby(group_cols):
        total_runs = len(group)
        successful_runs = int(group["success"].sum())
        success_rate = successful_runs / total_runs if total_runs > 0 else 0.0

        row = {
            "env_type": env_type,
            "algorithm": algorithm,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": total_runs - successful_runs,
            "success_rate": success_rate,
            "success_percent": success_rate * 100.0,
        }

        valid_group = group[group["success"] == True].copy()

        for metric in METRICS_TO_ANALYZE:
            if metric not in valid_group.columns:
                continue

            values = pd.to_numeric(valid_group[metric], errors="coerce").dropna()

            if len(values) == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_median"] = None
                continue

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
            row[f"{metric}_std"] = values.std()
            row[f"{metric}_median"] = values.median()

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="CSV de métricas acumuladas, por ejemplo env1_gbp_metrics.csv env1_orca_plus_metrics.csv"
    )
    parser.add_argument(
        "--output_all",
        default="results/analysis/comparison_all_runs.csv"
    )
    parser.add_argument(
        "--output_summary",
        default="results/analysis/comparison_summary.csv"
    )

    args = parser.parse_args()

    df = load_metrics_files(args.metrics)
    df = normalize_success_column(df)

    os.makedirs(os.path.dirname(args.output_all), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)

    df.to_csv(args.output_all, index=False)

    summary = build_summary(df)
    summary.to_csv(args.output_summary, index=False)

    print(f"CSV con todas las runs generado: {args.output_all}")
    print(f"CSV resumen generado: {args.output_summary}")


if __name__ == "__main__":
    main()