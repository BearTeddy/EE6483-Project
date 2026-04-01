from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelMetrics:
    model_name: str
    source_path: Path
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[float]]


def discover_metric_files(reports_dir: Path, models_dir: Path) -> list[Path]:
    files: list[Path] = []
    files.extend(sorted(reports_dir.glob("*_metrics.json")))
    files.extend(sorted(models_dir.glob("*/metrics.json")))
    return files


def dedupe_by_model(metrics: list[ModelMetrics]) -> list[ModelMetrics]:
    unique: dict[str, ModelMetrics] = {}
    for item in metrics:
        if item.model_name not in unique:
            unique[item.model_name] = item
    return list(unique.values())


def _infer_model_name(path: Path, payload: dict[str, Any]) -> str:
    if "model_name" in payload and payload["model_name"]:
        return str(payload["model_name"])

    if path.name == "metrics.json":
        return path.parent.name

    stem = path.stem
    if stem.endswith("_metrics"):
        return stem[: -len("_metrics")]
    return stem


def _normalize_cm(cm: list[list[Any]]) -> list[list[float]] | None:
    if not isinstance(cm, list) or len(cm) != 2:
        return None
    if not all(isinstance(row, list) and len(row) == 2 for row in cm):
        return None
    try:
        return [[float(cm[0][0]), float(cm[0][1])], [float(cm[1][0]), float(cm[1][1])]]
    except Exception:
        return None


def parse_metric_file(path: Path) -> ModelMetrics | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    model_name = _infer_model_name(path, payload)

    section = payload.get("best_validation", payload)
    accuracy = section.get("accuracy", payload.get("accuracy"))
    macro_f1 = section.get("macro_f1", payload.get("macro_f1"))
    cm = section.get("confusion_matrix", payload.get("confusion_matrix"))
    cm2 = _normalize_cm(cm)

    if accuracy is None or macro_f1 is None or cm2 is None:
        return None

    return ModelMetrics(
        model_name=model_name,
        source_path=path,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        confusion_matrix=cm2,
    )


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[c]) for c in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_summary(metrics: list[ModelMetrics], output_dir: Path) -> pd.DataFrame:
    rows = []
    for m in metrics:
        tn, fp = m.confusion_matrix[0]
        fn, tp = m.confusion_matrix[1]
        total = tn + fp + fn + tp
        rows.append(
            {
                "model_name": m.model_name,
                "accuracy": round(m.accuracy, 6),
                "macro_f1": round(m.macro_f1, 6),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "total": int(total),
                "source": str(m.source_path.relative_to(PROJECT_ROOT)),
            }
        )

    df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    (output_dir / "comparison_results.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "comparison_results.csv", index=False)
    (output_dir / "comparison_results.md").write_text(to_markdown_table(df), encoding="utf-8")
    return df


def plot_metrics_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(max(8, 1.3 * len(df)), 5))
    plot_df = df.melt(id_vars="model_name", value_vars=["accuracy", "macro_f1"], var_name="metric", value_name="value")
    ax = sns.barplot(data=plot_df, x="model_name", y="value", hue="metric")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison: Accuracy vs Macro-F1")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison_bar.png", dpi=160)
    plt.close()


def plot_confusion_per_model(metrics: list[ModelMetrics], output_dir: Path) -> None:
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for m in metrics:
        cm = m.confusion_matrix
        plt.figure(figsize=(4, 3.5))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(f"Confusion Matrix: {m.model_name}")
        plt.tight_layout()
        safe_name = m.model_name.replace("/", "_").replace("-", "_")
        plt.savefig(cm_dir / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close()


def plot_confusion_grid(metrics: list[ModelMetrics], output_dir: Path) -> None:
    n = len(metrics)
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.6))
    axes = np.array(axes, dtype=object).reshape(-1).tolist()

    for ax, m in zip(axes, metrics):
        sns.heatmap(
            m.confusion_matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            xticklabels=["P0", "P1"],
            yticklabels=["T0", "T1"],
            ax=ax,
        )
        ax.set_title(m.model_name)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices Across Models", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_comparison_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_comparison_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    compare = df.copy()
    compare["tn_rate"] = compare["tn"] / compare["total"]
    compare["fp_rate"] = compare["fp"] / compare["total"]
    compare["fn_rate"] = compare["fn"] / compare["total"]
    compare["tp_rate"] = compare["tp"] / compare["total"]
    matrix_df = compare.set_index("model_name")[["tn_rate", "fp_rate", "fn_rate", "tp_rate"]]

    plt.figure(figsize=(7.2, max(3.2, 0.5 * len(matrix_df) + 1.8)))
    ax = sns.heatmap(matrix_df, annot=True, fmt=".3f", cmap="viridis")
    ax.set_title("Confusion Pattern Comparison (Normalized)")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_pattern_comparison.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model comparison charts and confusion-matrix reports.")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = discover_metric_files(args.reports_dir, args.models_dir)
    parsed = [parse_metric_file(path) for path in metric_files]
    metrics = [m for m in parsed if m is not None]
    metrics = dedupe_by_model(metrics)

    if not metrics:
        raise SystemExit(
            "No valid metrics found. Expected files such as reports/*_metrics.json or models/*/metrics.json."
        )

    df = save_summary(metrics, args.output_dir)
    plot_metrics_comparison(df, args.output_dir)
    plot_confusion_per_model(metrics, args.output_dir)
    plot_confusion_grid(metrics, args.output_dir)
    plot_confusion_comparison_matrix(df, args.output_dir)

    print("Comparison report generated.")
    print(f"Output directory: {args.output_dir}")
    print(f"Models included: {len(metrics)}")
    print(f"Summary CSV: {args.output_dir / 'comparison_results.csv'}")
    print(f"Summary Markdown: {args.output_dir / 'comparison_results.md'}")


if __name__ == "__main__":
    main()
