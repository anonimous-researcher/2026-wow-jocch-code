#!/usr/bin/env python3
"""
Plot stacked correlations between:
  - gt_num_entailed
  - text length
  - mean_accuracy_across_models (Graph 1)
  - mean_f_score_across_models (Graph 2)

Input: one of the aggregated CSVs produced by your analysis pipeline.
Output: two PDFs saved to outputs/analysis/graphs/

Usage examples:
  python plot_text_correlations.py --input outputs/analysis/text_aggregate_metrics_all_xlang.csv
  python plot_text_correlations.py --input outputs/analysis/text_aggregate_metrics_c.csv --smooth-window 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/analysis/text_aggregate_metrics_all_xlang.csv", help="Path to one generated metrics CSV.")
    p.add_argument(
        "--out_dir",
        default="outputs/analysis/graphs",
        help="Destination folder for PDF graphs.",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Optional rolling-median window size (0 disables smoothing).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster DPI (only affects embedded raster elements; PDFs are vector by default).",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def robust_text_length(s: str) -> int:
    if s is None:
        return 0
    if isinstance(s, float) and np.isnan(s):
        return 0
    s = str(s)
    return len(s)


def rolling_median(series: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return series
    # center=True gives nicer visuals for "trend" overlays
    return series.rolling(window=window, center=True, min_periods=max(3, window // 3)).median()


def make_stacked_plot(
    df: pd.DataFrame,
    sort_col: str,
    metric_label: str,
    metric_color: str,
    gt_color: str,
    len_color: str,
    out_path: Path,
    smooth_window: int = 0,
    title_prefix: str = "",
) -> None:
    # Sort by chosen metric
    d = df.sort_values(sort_col, ascending=True).reset_index(drop=True)

    x = np.arange(len(d))
    gt = d["gt_num_entailed"].astype(float)
    tl = d["text_length"].astype(float)
    metric = d[sort_col].astype(float)

    # Optional smoothing overlay
    gt_sm = rolling_median(gt, smooth_window)
    tl_sm = rolling_median(tl, smooth_window)
    metric_sm = rolling_median(metric, smooth_window)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(11, 9),
        gridspec_kw={"hspace": 0.12},
    )

    # --- Plot 1: gt_num_entailed
    ax = axes[0]
    ax.plot(x, gt, color=gt_color, alpha=0.35, linewidth=1.0, label="# of entailed labels (raw)")
    if smooth_window and smooth_window > 1:
        ax.plot(x, gt_sm, color=gt_color, alpha=0.95, linewidth=2.0, label=f"rolling median (w={smooth_window})")
    ax.set_ylabel("# of entailed labels")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    # --- Plot 2: text_length
    ax = axes[1]
    ax.plot(x, tl, color=len_color, alpha=0.35, linewidth=1.0, label="text length (raw)")
    if smooth_window and smooth_window > 1:
        ax.plot(x, tl_sm, color=len_color, alpha=0.95, linewidth=2.0, label=f"rolling median (w={smooth_window})")
    ax.set_ylabel("text length (chars)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    # --- Plot 3: metric
    ax = axes[2]
    ax.plot(x, metric, color=metric_color, alpha=0.35, linewidth=1.0, label=f"{metric_label} (raw)")
    if smooth_window and smooth_window > 1:
        ax.plot(x, metric_sm, color=metric_color, alpha=0.95, linewidth=2.0, label=f"rolling median (w={smooth_window})")
    ax.set_ylabel(metric_label)
    ax.set_xlabel(f"texts sorted by {metric_label} (ascending)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    # Title with quick dataset info
    n = len(d)
    metric_min = float(np.nanmin(metric)) if n else float("nan")
    metric_max = float(np.nanmax(metric)) if n else float("nan")
    # fig.suptitle(
    #     f"{title_prefix}{metric_label} vs gt_num_entailed and text_length  "
    #     f"(N={n}, {metric_label} range: {metric_min:.3f}–{metric_max:.3f})",
    #     y=0.99,
    #     fontsize=14,
    # )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    # Required columns for your two plots
    require_columns(
        df,
        ["text", "gt_num_entailed", "mean_accuracy_across_models", "mean_f_score_across_models"],
        in_path,
    )

    # Compute text length (characters)
    df = df.copy()
    df["text_length"] = df["text"].apply(robust_text_length)

    # Ensure numeric
    for c in ["gt_num_entailed", "mean_accuracy_across_models", "mean_f_score_across_models"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["text_length"] = pd.to_numeric(df["text_length"], errors="coerce")

    # Drop rows without the sorting metric (keeps plots clean)
    df_acc = df.dropna(subset=["mean_accuracy_across_models"]).copy()
    df_f1 = df.dropna(subset=["mean_f_score_across_models"]).copy()

    # Colors (consistent across both graphs for gt_num_entailed and text_length)
    # Chosen for readability and print friendliness.
    GT_COLOR = "#D55E00"   # vermillion/red-orange
    LEN_COLOR = "#0072B2"  # blue
    ACC_COLOR = "#009E73"  # green
    F1_COLOR  = "#CC79A7"  # purple/pink

    stem = in_path.stem  # e.g., text_aggregate_metrics_all_xlang
    out1 = out_dir / f"aggregate_accuracy.pdf"
    out2 = out_dir / f"aggregate_fscore.pdf"

    make_stacked_plot(
        df=df_acc,
        sort_col="mean_accuracy_across_models",
        metric_label="mean accuracy across models",
        metric_color=ACC_COLOR,
        gt_color=GT_COLOR,
        len_color=LEN_COLOR,
        out_path=out1,
        smooth_window=args.smooth_window,
        title_prefix="Graph 1 — ",
    )

    make_stacked_plot(
        df=df_f1,
        sort_col="mean_f_score_across_models",
        metric_label="mean f-score across models",
        metric_color=F1_COLOR,
        gt_color=GT_COLOR,
        len_color=LEN_COLOR,
        out_path=out2,
        smooth_window=args.smooth_window,
        title_prefix="Graph 2 — ",
    )

    print(f"Wrote:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()
