#!/usr/bin/env python3
"""
Compute confusion-matrix counts (TP/TN/FP/FN) per keyword in each keyword set.

Assumptions (matching your aggregation script):
- Predictions live in outputs/raw_results/
  quotes_<keyword_set>_<language>_<model>_<template>.csv
    keyword_set in {c,k}
    language in {en,en_nmt,it}
    model in {M1..M8}
    template in {T0,T1}
    columns: text,time,<kw1>,...,<kwN>

- Ground truth live in data/
  quotes_<keyword_set>_<language>.csv
    columns: text,<kw1>,...,<kwN>

Keyword identity across languages is by COLUMN ORDER within the same keyword_set:
  kw_id = 0..K-1

We also attach a canonical EN label per (keyword_set, kw_id) from EN GT column names.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PRED_RE = re.compile(
    r"^quotes_(?P<keyword_set>c|k)_(?P<language>en|en_nmt|it)_(?P<model>M[1-8])_(?P<template>T[01])\.csv$"
)
GT_RE = re.compile(
    r"^quotes_(?P<keyword_set>c|k)_(?P<language>en|en_nmt|it)\.csv$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute TP/TN/FP/FN per keyword per keyword_set.")
    p.add_argument("--pred_dir", type=str, default="outputs/raw_results", help="Folder with prediction CSV files.")
    p.add_argument("--gt_dir", type=str, default="data", help="Folder with ground-truth CSV files.")
    p.add_argument("--out_dir", type=str, default="outputs/analysis", help="Destination folder for output CSVs.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for y_pred -> predicted label.")
    p.add_argument(
        "--strict_keywords",
        action="store_true",
        help=(
            "If set, enforce same number of keyword columns between GT and predictions for each (set,lang). "
            "Keyword NAMES may differ; only counts/order matter."
        ),
    )
    p.add_argument(
        "--breakdown",
        choices=["none", "language", "model", "template", "language_model", "language_model_template"],
        default="none",
        help="Optional breakdown level for counts.",
    )
    p.add_argument(
        "--also_write_global_all",
        action="store_true",
        help="Also write an 'all' file that treats keywords from c and k as distinct via kw_global_id = <set>::<kw_id>.",
    )
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).replace({"": np.nan})


def to_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_gt_index(gt_dir: Path) -> Dict[Tuple[str, str], Path]:
    index: Dict[Tuple[str, str], Path] = {}
    for p in gt_dir.iterdir():
        if not p.is_file():
            continue
        m = GT_RE.match(p.name)
        if m:
            index[(m.group("keyword_set"), m.group("language"))] = p
    return index


def melt_ground_truth_with_rowid_and_kwid(gt_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns:
      gt_long columns: [row_id, kw_id, y_true]
      kw_cols_order: keyword column names in order
    """
    gt = safe_read_csv(gt_path)
    if "text" not in gt.columns:
        raise ValueError(f"{gt_path} missing required column 'text'")

    gt = gt.reset_index(drop=False).rename(columns={"index": "row_id"})
    kw_cols = [c for c in gt.columns if c not in ("row_id", "text")]
    if not kw_cols:
        raise ValueError(f"{gt_path} has no keyword columns (expected after text)")

    gt = to_numeric_cols(gt, kw_cols)

    long = gt.melt(
        id_vars=["row_id"],
        value_vars=kw_cols,
        var_name="kw_colname",
        value_name="y_true",
    )
    col_to_id = {c: i for i, c in enumerate(kw_cols)}
    long["kw_id"] = long["kw_colname"].map(col_to_id).astype(int)
    long = long.drop(columns=["kw_colname"])

    return long, kw_cols


def melt_predictions_with_rowid_and_kwid(pred_path: Path, meta: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns:
      pred_long columns:
        [row_id, time, kw_id, y_pred, keyword_set, language, model, template]
      kw_cols_order: keyword column names in order
    """
    df = safe_read_csv(pred_path)
    if "text" not in df.columns:
        raise ValueError(f"{pred_path} missing required column 'text'")
    if "time" not in df.columns:
        raise ValueError(f"{pred_path} missing required column 'time'")

    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    kw_cols = [c for c in df.columns if c not in ("row_id", "text", "time")]
    if not kw_cols:
        raise ValueError(f"{pred_path} has no keyword columns (expected after text,time)")

    df = to_numeric_cols(df, ["time"] + kw_cols)

    long = df.melt(
        id_vars=["row_id", "time"],
        value_vars=kw_cols,
        var_name="kw_colname",
        value_name="y_pred",
    )
    col_to_id = {c: i for i, c in enumerate(kw_cols)}
    long["kw_id"] = long["kw_colname"].map(col_to_id).astype(int)
    long = long.drop(columns=["kw_colname"])

    for k, v in meta.items():
        long[k] = v

    return long, kw_cols


def build_kw_label_en(gt_index: Dict[Tuple[str, str], Path]) -> Dict[Tuple[str, int], str]:
    """
    (keyword_set, kw_id) -> kw_label_en (from EN GT column names)
    """
    out: Dict[Tuple[str, int], str] = {}
    for ks in ["c", "k"]:
        key = (ks, "en")
        if key not in gt_index:
            raise FileNotFoundError(f"Missing canonical English ground truth: {gt_index.get(key, Path(f'quotes_{ks}_en.csv'))}")
        gt_en = safe_read_csv(gt_index[key])
        if "text" not in gt_en.columns:
            raise ValueError(f"{gt_index[key]} missing required column 'text'")
        kw_cols = [c for c in gt_en.columns if c != "text"]
        if not kw_cols:
            raise ValueError(f"{gt_index[key]} has no keyword columns")
        for i, col in enumerate(kw_cols):
            out[(ks, i)] = col
    return out


def breakdown_group_cols(breakdown: str) -> List[str]:
    if breakdown == "none":
        return ["keyword_set", "kw_id"]
    if breakdown == "language":
        return ["keyword_set", "language", "kw_id"]
    if breakdown == "model":
        return ["keyword_set", "model", "kw_id"]
    if breakdown == "template":
        return ["keyword_set", "template", "kw_id"]
    if breakdown == "language_model":
        return ["keyword_set", "language", "model", "kw_id"]
    if breakdown == "language_model_template":
        return ["keyword_set", "language", "model", "template", "kw_id"]
    raise ValueError(f"Unknown breakdown: {breakdown}")


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    gt_index = build_gt_index(gt_dir)
    if not gt_index:
        raise FileNotFoundError(
            f"No ground-truth files found in {gt_dir}. Expected names like quotes_c_en.csv, quotes_k_it.csv, ..."
        )

    # Canonical EN keyword labels by position
    kw_label_en = build_kw_label_en(gt_index)

    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file() and PRED_RE.match(p.name)])
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}. Expected names like quotes_c_en_M1_T0.csv")

    # Cache GT longs by (keyword_set, language) and keyword counts
    gt_long_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    gt_kwcount_cache: Dict[Tuple[str, str], int] = {}

    all_rows: List[pd.DataFrame] = []
    skipped: List[str] = []
    thr = float(args.threshold)

    for pf in pred_files:
        m = PRED_RE.match(pf.name)
        assert m is not None
        meta = m.groupdict()
        ks, lang = meta["keyword_set"], meta["language"]
        key = (ks, lang)

        if key not in gt_index:
            skipped.append(f"{pf.name} (missing GT: {gt_dir / f'quotes_{ks}_{lang}.csv'})")
            continue

        if key not in gt_long_cache:
            gt_long, gt_kw_cols = melt_ground_truth_with_rowid_and_kwid(gt_index[key])
            gt_long_cache[key] = gt_long
            gt_kwcount_cache[key] = len(gt_kw_cols)

        pred_long, pred_kw_cols = melt_predictions_with_rowid_and_kwid(pf, meta)

        if args.strict_keywords and len(pred_kw_cols) != gt_kwcount_cache[key]:
            raise ValueError(
                f"Keyword column count mismatch for {pf.name} (pred={len(pred_kw_cols)}) "
                f"vs GT {gt_index[key].name} (gt={gt_kwcount_cache[key]})."
            )

        gt_long = gt_long_cache[key]

        merged = pred_long.merge(gt_long[["row_id", "kw_id", "y_true"]], on=["row_id", "kw_id"], how="inner")
        merged = merged.dropna(subset=["y_true", "y_pred", "time"])
        if merged.empty:
            skipped.append(f"{pf.name} (merge empty after aligning with {gt_index[key].name})")
            continue

        merged["row_id"] = merged["row_id"].astype(int)
        merged["kw_id"] = merged["kw_id"].astype(int)
        merged["y_true"] = pd.to_numeric(merged["y_true"], errors="coerce")
        merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="coerce")
        merged["time"] = pd.to_numeric(merged["time"], errors="coerce")
        merged = merged.dropna(subset=["y_true", "y_pred", "time"])

        merged["pred_label"] = (merged["y_pred"] >= thr).astype(int)
        y_true = merged["y_true"].astype(int)

        merged["tp"] = ((merged["pred_label"] == 1) & (y_true == 1)).astype(int)
        merged["tn"] = ((merged["pred_label"] == 0) & (y_true == 0)).astype(int)
        merged["fp"] = ((merged["pred_label"] == 1) & (y_true == 0)).astype(int)
        merged["fn"] = ((merged["pred_label"] == 0) & (y_true == 1)).astype(int)

        # Attach canonical EN label by (set, kw_id)
        merged["kw_label_en"] = merged.apply(lambda r: kw_label_en.get((r["keyword_set"], int(r["kw_id"])), ""), axis=1)

        # Optional: global id/label to keep c and k disjoint in an "all" table
        merged["kw_global_id"] = merged["keyword_set"].astype(str) + "::" + merged["kw_id"].astype(str)
        merged["kw_global_label_en"] = merged["keyword_set"].astype(str) + "::" + merged["kw_label_en"].astype(str)

        all_rows.append(merged)

    if not all_rows:
        raise RuntimeError("No usable data after merging predictions with ground truth.")

    data = pd.concat(all_rows, ignore_index=True)

    # --- Main per-keyword per-set table (with optional breakdowns)
    group_cols = breakdown_group_cols(args.breakdown)
    agg = (
        data.groupby(group_cols, dropna=False)
        .agg(
            kw_label_en=("kw_label_en", "first"),
            n_pairs=("kw_id", "size"),
            tp=("tp", "sum"),
            tn=("tn", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
        )
        .reset_index()
    )

    # Add derived rates (useful, but you can drop if you want pure counts)
    tp = agg["tp"].astype(float)
    fp = agg["fp"].astype(float)
    fn = agg["fn"].astype(float)
    tn = agg["tn"].astype(float)
    agg["support_pos"] = tp + fn
    agg["support_neg"] = tn + fp
    agg["precision"] = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    agg["recall"] = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    agg["f1"] = np.where(agg["precision"] + agg["recall"] > 0, 2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"]), 0.0)

    # Sort: worst by fp then fn (common diagnostic), then by support
    agg = agg.sort_values(["keyword_set", "fp", "fn", "support_pos"], ascending=[True, False, False, False])

    out_main = out_dir / f"keyword_confusion_counts_{args.breakdown}.csv"
    agg.to_csv(out_main, index=False)

    # --- Optional: a single "all" table where c/k keywords are disjoint (kw_global_id)
    if args.also_write_global_all:
        group_cols_all = [c for c in group_cols if c != "kw_id"]
        # replace kw_id with kw_global_id
        group_cols_all = [c for c in group_cols_all if c not in ("keyword_set",)]
        # keep keyword_set in grouping only if it is in breakdown; otherwise "all" means collapse sets.
        # we want disjoint ids anyway, so we can safely drop keyword_set and use kw_global_id.
        group_cols_all = [c for c in group_cols_all if c != "keyword_set"] + ["kw_global_id"]
        # attach label
        agg_all = (
            data.groupby(group_cols_all, dropna=False)
            .agg(
                kw_label_en=("kw_global_label_en", "first"),
                n_pairs=("kw_global_id", "size"),
                tp=("tp", "sum"),
                tn=("tn", "sum"),
                fp=("fp", "sum"),
                fn=("fn", "sum"),
            )
            .reset_index()
        )
        agg_all.to_csv(out_dir / f"keyword_confusion_counts_all_{args.breakdown}.csv", index=False)

    if skipped:
        (out_dir / "keyword_confusion_counts_skipped_files.txt").write_text("\n".join(skipped), encoding="utf-8")

    print(f"Wrote: {out_main}")
    if args.also_write_global_all:
        print(f"Wrote: {out_dir / f'keyword_confusion_counts_all_{args.breakdown}.csv'}")
    if skipped:
        print(f"Skipped files logged in: {out_dir / 'keyword_confusion_counts_skipped_files.txt'}")


if __name__ == "__main__":
    main()
