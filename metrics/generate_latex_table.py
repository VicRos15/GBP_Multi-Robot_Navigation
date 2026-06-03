#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import pandas as pd


ALGORITHM_LABELS = {
    "gbp": "GBP",
    "orca_plus": "ORCA+",
    "pred_dwa": "Predictive DWA",
}


def fmt_mean_std(mean, std, decimals=2):
    if pd.isna(mean):
        return "--"

    if pd.isna(std):
        return f"{mean:.{decimals}f}"

    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def fmt_percent(value):
    if pd.isna(value):
        return "--"

    return f"{value:.1f}"


def generate_latex_table(df: pd.DataFrame, env_name: str) -> str:
    lines = []

    env_label = env_name.lower()

    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Performance comparison for {env_label}.}}")
    lines.append(f"\\label{{tab:{env_label}_results}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")

    # Solo una línea vertical entre Method y las métricas
    lines.append("\\begin{tabular}{l|ccccc}")
    lines.append("\\toprule")

    # Cabecera simple, sin Robustness/Efficiency/Safety/Smoothness
    lines.append(
        "Method & "
        "Success [\\%] $\\uparrow$ & "
        "Makespan [s] $\\downarrow$ & "
        "Distance [cm] $\\downarrow$ & "
        "Safety [cm] $\\uparrow$ & "
        "LDJ $\\downarrow$ \\\\"
    )

    lines.append("\\midrule")

    preferred_order = ["gbp", "orca_plus", "pred_dwa"]

    for algorithm in preferred_order:
        row = df[df["algorithm"] == algorithm]

        if row.empty:
            continue

        row = row.iloc[0]

        method = ALGORITHM_LABELS.get(algorithm, algorithm)

        success = fmt_percent(row.get("success_percent"))

        makespan = fmt_mean_std(
            row.get("makespan_sim_mean"),
            row.get("makespan_sim_std"),
            decimals=2
        )

        distance = fmt_mean_std(
            row.get("total_distance_mean"),
            row.get("total_distance_std"),
            decimals=1
        )

        safety = fmt_mean_std(
            row.get("mean_safety_clearance_mean"),
            row.get("mean_safety_clearance_std"),
            decimals=1
        )

        smoothness = fmt_mean_std(
            row.get("mean_smoothness_ldj_mean"),
            row.get("mean_smoothness_ldj_std"),
            decimals=2
        )

        lines.append(
            f"{method} & "
            f"{success} & "
            f"{makespan} & "
            f"{distance} & "
            f"{safety} & "
            f"{smoothness} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        required=True,
        help="CSV resumen generado por build_comparison.py"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Archivo .tex de salida"
    )
    parser.add_argument(
        "--env_name",
        default=None,
        help="Nombre del entorno para caption/label. Si no se indica, se toma del CSV."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.summary)

    if args.env_name is not None:
        env_name = args.env_name
    elif "env_type" in df.columns and len(df["env_type"].dropna()) > 0:
        env_name = str(df["env_type"].dropna().iloc[0])
    else:
        env_name = "environment"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    latex = generate_latex_table(df, env_name)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Tabla LaTeX generada: {args.output}")


if __name__ == "__main__":
    main()