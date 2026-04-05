from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentiment_project.core import (
    build_run_label,
    format_test_size_label,
    format_test_size_tag,
    sanitize_name,
)


@dataclass
class RunMetrics:
    model_name: str
    source_path: Path
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[float]]
    test_size: float
    seed: int | None
    train_samples: int | None
    validation_samples: int | None
    num_samples: int | None
    fit_seconds: float | None
    eval_seconds: float | None
    total_seconds: float | None

    @property
    def comparison_label(self) -> str:
        return f"{self.model_name} | test={format_test_size_label(self.test_size)}"

    @property
    def run_label(self) -> str:
        return build_run_label(self.model_name, self.test_size, self.seed)


@dataclass
class AggregateMetrics:
    model_name: str
    test_size: float
    seeds: list[int]
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: list[list[float]]
    train_samples_mean: float
    validation_samples_mean: float
    fit_seconds_mean: float
    eval_seconds_mean: float
    total_seconds_mean: float

    @property
    def comparison_label(self) -> str:
        return f"{self.model_name} | test={format_test_size_label(self.test_size)}"

    @property
    def run_count(self) -> int:
        return len(self.seeds)


def discover_metric_files(reports_dir: Path, models_dir: Path) -> list[Path]:
    discovered: set[Path] = set()
    patterns = [
        ("reports", reports_dir, "*_metrics.json"),
        ("reports", reports_dir, "metrics.json"),
        ("models", models_dir, "metrics.json"),
    ]

    for _, root, pattern in patterns:
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            if path.is_file():
                discovered.add(path.resolve())

    return sorted(discovered)


def _infer_model_name(path: Path, payload: dict[str, Any]) -> str:
    if "model_name" in payload and payload["model_name"]:
        return str(payload["model_name"])

    parts = list(path.parts)
    if "experiments" in parts:
        experiments_index = parts.index("experiments")
        if experiments_index + 1 < len(parts):
            return parts[experiments_index + 1]

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


def _extract_number(payload: dict[str, Any], *candidates: Any, cast=float) -> Any:
    for value in candidates:
        if value is None:
            continue
        try:
            return cast(value)
        except Exception:
            continue
    return None


def parse_metric_file(path: Path) -> RunMetrics | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    model_name = _infer_model_name(path, payload)
    section = payload.get("best_validation", payload)
    settings = payload.get("settings", {})
    split_metadata = payload.get("split_metadata", {})
    timing = payload.get("timing", {})

    accuracy = _extract_number(section, section.get("accuracy"), payload.get("accuracy"))
    macro_f1 = _extract_number(section, section.get("macro_f1"), payload.get("macro_f1"))
    cm = section.get("confusion_matrix", payload.get("confusion_matrix"))
    cm2 = _normalize_cm(cm)

    test_size = _extract_number(
        payload,
        payload.get("test_size"),
        split_metadata.get("test_size"),
        settings.get("test_size"),
    )
    seed = _extract_number(
        payload,
        payload.get("seed"),
        split_metadata.get("seed"),
        settings.get("seed"),
        settings.get("random_state"),
        cast=int,
    )
    train_samples = _extract_number(
        payload,
        payload.get("train_samples"),
        split_metadata.get("train_samples"),
        cast=int,
    )
    validation_samples = _extract_number(
        payload,
        payload.get("validation_samples"),
        split_metadata.get("validation_samples"),
        cast=int,
    )
    num_samples = _extract_number(
        payload,
        payload.get("num_samples"),
        split_metadata.get("num_samples"),
        cast=int,
    )

    if accuracy is None or macro_f1 is None or cm2 is None or test_size is None:
        return None

    return RunMetrics(
        model_name=model_name,
        source_path=path,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        confusion_matrix=cm2,
        test_size=float(test_size),
        seed=int(seed) if seed is not None else None,
        train_samples=int(train_samples) if train_samples is not None else None,
        validation_samples=int(validation_samples) if validation_samples is not None else None,
        num_samples=int(num_samples) if num_samples is not None else None,
        fit_seconds=_extract_number(timing, timing.get("fit_seconds")),
        eval_seconds=_extract_number(timing, timing.get("eval_seconds")),
        total_seconds=_extract_number(timing, timing.get("total_seconds")),
    )


def _priority_for_path(path: Path) -> tuple[int, int, str]:
    try:
        relative = path.relative_to(PROJECT_ROOT)
        rel_str = str(relative)
    except Exception:
        rel_str = str(path)

    priority = 0 if rel_str.startswith("reports/") else 1
    return (priority, len(rel_str), rel_str)


def dedupe_by_run(metrics: list[RunMetrics]) -> list[RunMetrics]:
    unique: dict[tuple[str, float, int | None], RunMetrics] = {}
    for item in metrics:
        key = (item.model_name, round(item.test_size, 8), item.seed)
        current = unique.get(key)
        if current is None or _priority_for_path(item.source_path) < _priority_for_path(current.source_path):
            unique[key] = item
    return list(unique.values())


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


def save_run_summary(metrics: list[RunMetrics], output_dir: Path) -> pd.DataFrame:
    rows = []
    for m in metrics:
        tn, fp = m.confusion_matrix[0]
        fn, tp = m.confusion_matrix[1]
        total = tn + fp + fn + tp
        rows.append(
            {
                "model_name": m.model_name,
                "test_size": round(m.test_size, 6),
                "test_size_label": format_test_size_label(m.test_size),
                "seed": "" if m.seed is None else int(m.seed),
                "accuracy": round(m.accuracy, 6),
                "macro_f1": round(m.macro_f1, 6),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "total": int(total),
                "train_samples": "" if m.train_samples is None else int(m.train_samples),
                "validation_samples": "" if m.validation_samples is None else int(m.validation_samples),
                "fit_seconds": "" if m.fit_seconds is None else round(m.fit_seconds, 4),
                "eval_seconds": "" if m.eval_seconds is None else round(m.eval_seconds, 4),
                "total_seconds": "" if m.total_seconds is None else round(m.total_seconds, 4),
                "source": str(m.source_path.relative_to(PROJECT_ROOT)),
            }
        )

    df = pd.DataFrame(rows).sort_values(["model_name", "test_size", "seed"], na_position="last").reset_index(drop=True)
    df.to_csv(output_dir / "comparison_runs.csv", index=False)
    return df


def aggregate_metrics(runs: list[RunMetrics]) -> list[AggregateMetrics]:
    grouped: dict[tuple[str, float], list[RunMetrics]] = {}
    for run in runs:
        grouped.setdefault((run.model_name, run.test_size), []).append(run)

    aggregates: list[AggregateMetrics] = []
    for (model_name, test_size), group in grouped.items():
        confusion_sum = np.sum([np.array(item.confusion_matrix, dtype=float) for item in group], axis=0)
        accuracy_values = np.array([item.accuracy for item in group], dtype=float)
        macro_f1_values = np.array([item.macro_f1 for item in group], dtype=float)
        train_samples = np.array([item.train_samples for item in group if item.train_samples is not None], dtype=float)
        validation_samples = np.array(
            [item.validation_samples for item in group if item.validation_samples is not None],
            dtype=float,
        )
        fit_seconds = np.array([item.fit_seconds for item in group if item.fit_seconds is not None], dtype=float)
        eval_seconds = np.array([item.eval_seconds for item in group if item.eval_seconds is not None], dtype=float)
        total_seconds = np.array([item.total_seconds for item in group if item.total_seconds is not None], dtype=float)
        seeds = sorted(item.seed for item in group if item.seed is not None)

        aggregates.append(
            AggregateMetrics(
                model_name=model_name,
                test_size=test_size,
                seeds=seeds,
                accuracy_mean=float(accuracy_values.mean()),
                accuracy_std=float(accuracy_values.std(ddof=0)),
                macro_f1_mean=float(macro_f1_values.mean()),
                macro_f1_std=float(macro_f1_values.std(ddof=0)),
                confusion_matrix=confusion_sum.tolist(),
                train_samples_mean=float(train_samples.mean()) if train_samples.size else float("nan"),
                validation_samples_mean=float(validation_samples.mean()) if validation_samples.size else float("nan"),
                fit_seconds_mean=float(fit_seconds.mean()) if fit_seconds.size else float("nan"),
                eval_seconds_mean=float(eval_seconds.mean()) if eval_seconds.size else float("nan"),
                total_seconds_mean=float(total_seconds.mean()) if total_seconds.size else float("nan"),
            )
        )

    return sorted(aggregates, key=lambda item: (item.macro_f1_mean, item.accuracy_mean), reverse=True)


def save_aggregate_summary(metrics: list[AggregateMetrics], output_dir: Path) -> pd.DataFrame:
    rows = []
    for m in metrics:
        tn, fp = m.confusion_matrix[0]
        fn, tp = m.confusion_matrix[1]
        total = tn + fp + fn + tp
        rows.append(
            {
                "model_name": m.model_name,
                "test_size": round(m.test_size, 6),
                "test_size_label": format_test_size_label(m.test_size),
                "run_count": int(m.run_count),
                "seeds": ",".join(str(seed) for seed in m.seeds),
                "accuracy_mean": round(m.accuracy_mean, 6),
                "accuracy_std": round(m.accuracy_std, 6),
                "macro_f1_mean": round(m.macro_f1_mean, 6),
                "macro_f1_std": round(m.macro_f1_std, 6),
                "train_samples_mean": round(m.train_samples_mean, 2) if not math.isnan(m.train_samples_mean) else "",
                "validation_samples_mean": round(m.validation_samples_mean, 2)
                if not math.isnan(m.validation_samples_mean)
                else "",
                "fit_seconds_mean": round(m.fit_seconds_mean, 4) if not math.isnan(m.fit_seconds_mean) else "",
                "eval_seconds_mean": round(m.eval_seconds_mean, 4) if not math.isnan(m.eval_seconds_mean) else "",
                "total_seconds_mean": round(m.total_seconds_mean, 4) if not math.isnan(m.total_seconds_mean) else "",
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "total": int(total),
                "tn_rate": round((tn / total) if total else 0.0, 6),
                "fp_rate": round((fp / total) if total else 0.0, 6),
                "fn_rate": round((fn / total) if total else 0.0, 6),
                "tp_rate": round((tp / total) if total else 0.0, 6),
                "comparison_label": m.comparison_label,
            }
        )

    df = pd.DataFrame(rows).sort_values(["macro_f1_mean", "accuracy_mean"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(output_dir / "comparison_results.csv", index=False)
    (output_dir / "comparison_results.md").write_text(to_markdown_table(df), encoding="utf-8")
    return df


def _annotated_matrix(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = cm.sum()
    normalized = cm / total if total else np.zeros_like(cm, dtype=float)
    annotations = np.array(
        [[f"{int(count)}\n({rate:.3f})" for count, rate in zip(count_row, rate_row)] for count_row, rate_row in zip(cm, normalized)],
        dtype=object,
    )
    return normalized, annotations


def plot_metrics_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(max(8, 1.4 * len(df)), 5))
    plot_df = df.melt(
        id_vars="comparison_label",
        value_vars=["accuracy_mean", "macro_f1_mean"],
        var_name="metric",
        value_name="value",
    )
    plot_df["metric"] = plot_df["metric"].map({"accuracy_mean": "accuracy", "macro_f1_mean": "macro_f1"})
    ax = sns.barplot(data=plot_df, x="comparison_label", y="value", hue="metric")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison by Holdout Size")
    ax.set_xlabel("Model / Test Size")
    ax.set_ylabel("Mean Score")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison_bar.png", dpi=160)
    plt.close()


def plot_confusion_per_run(metrics: list[RunMetrics], output_dir: Path) -> None:
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for m in metrics:
        cm = np.array(m.confusion_matrix, dtype=float)
        plt.figure(figsize=(4.6, 3.8))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(m.run_label)
        plt.tight_layout()
        filename = (
            f"confusion_matrix_{sanitize_name(m.model_name)}_test_{format_test_size_tag(m.test_size)}"
            f"_seed_{m.seed if m.seed is not None else 'na'}.png"
        )
        plt.savefig(cm_dir / filename, dpi=160)
        plt.close()


def plot_confusion_per_aggregate(metrics: list[AggregateMetrics], output_dir: Path) -> None:
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for m in metrics:
        cm = np.array(m.confusion_matrix, dtype=float)
        normalized, annotations = _annotated_matrix(cm)
        plt.figure(figsize=(4.8, 4.0))
        ax = sns.heatmap(
            normalized,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=False,
            vmin=0.0,
            vmax=1.0,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(f"{m.comparison_label} | aggregate")
        plt.tight_layout()
        filename = f"confusion_matrix_{sanitize_name(m.model_name)}_test_{format_test_size_tag(m.test_size)}_aggregate.png"
        plt.savefig(cm_dir / filename, dpi=160)
        plt.close()


def plot_confusion_grid(metrics: list[AggregateMetrics], output_dir: Path) -> None:
    n = len(metrics)
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.0))
    axes = np.array(axes, dtype=object).reshape(-1).tolist()

    for ax, m in zip(axes, metrics):
        cm = np.array(m.confusion_matrix, dtype=float)
        normalized, _ = _annotated_matrix(cm)
        sns.heatmap(
            normalized,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            cbar=False,
            vmin=0.0,
            vmax=1.0,
            xticklabels=["P0", "P1"],
            yticklabels=["T0", "T1"],
            ax=ax,
        )
        ax.set_title(m.comparison_label)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.suptitle("Normalized Confusion Matrices Across Model / Split Combinations", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_comparison_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_comparison_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    matrix_df = df.set_index("comparison_label")[["tn_rate", "fp_rate", "fn_rate", "tp_rate"]]

    plt.figure(figsize=(7.2, max(3.2, 0.5 * len(matrix_df) + 1.8)))
    ax = sns.heatmap(matrix_df, annot=True, fmt=".3f", cmap="viridis")
    ax.set_title("Normalized Confusion Pattern Comparison")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Model / Test Size")
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
    if not metric_files:
        raise SystemExit(
            "No metrics discovered. Expected files such as reports/**/*_metrics.json, reports/**/metrics.json, or models/**/metrics.json."
        )

    parsed: list[RunMetrics] = []
    for path in tqdm(metric_files, desc="Parsing metric files", unit="file"):
        metric = parse_metric_file(path)
        if metric is not None:
            parsed.append(metric)

    metrics = dedupe_by_run(parsed)
    if not metrics:
        raise SystemExit(
            "No valid metrics found. Expected accuracy, macro_f1, confusion_matrix, and test_size fields in metrics JSON."
        )

    run_df = save_run_summary(metrics, args.output_dir)
    aggregate_metrics_list = aggregate_metrics(metrics)
    aggregate_df = save_aggregate_summary(aggregate_metrics_list, args.output_dir)

    steps = [
        ("metrics bar chart", lambda: plot_metrics_comparison(aggregate_df, args.output_dir)),
        ("per-run confusion matrices", lambda: plot_confusion_per_run(metrics, args.output_dir)),
        ("aggregate confusion matrices", lambda: plot_confusion_per_aggregate(aggregate_metrics_list, args.output_dir)),
        ("confusion grid", lambda: plot_confusion_grid(aggregate_metrics_list, args.output_dir)),
        ("confusion pattern heatmap", lambda: plot_confusion_comparison_matrix(aggregate_df, args.output_dir)),
    ]
    for label, action in tqdm(steps, desc="Generating report assets", unit="step"):
        action()

    print("Comparison report generated.")
    print(f"Output directory: {args.output_dir}")
    print(f"Runs included: {len(run_df)}")
    print(f"Aggregate rows: {len(aggregate_df)}")
    print(f"Raw runs CSV: {args.output_dir / 'comparison_runs.csv'}")
    print(f"Aggregate CSV: {args.output_dir / 'comparison_results.csv'}")
    print(f"Aggregate Markdown: {args.output_dir / 'comparison_results.md'}")


if __name__ == "__main__":
    main()
