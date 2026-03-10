#!/usr/bin/env python3
"""
Aggregate entailment experiment CSVs into per-text metrics CSVs.

Key features:
- Predictions in outputs/raw_results/
- Ground truth in data/
- Rows are aligned by order across languages (row_id)
- Keywords are aligned across languages by COLUMN ORDER within the same keyword_set:
    kw_id = 0..K-1 (excluding text/time)
  This avoids double-counting translated keyword names across languages.
- Duplicate texts are handled by using row_id-based keys (not text strings).
- Produces split outputs (c/k) and true aggregated outputs (all) that collapse across keyword sets.
- Adds gt_num_entailed and mean_f_score_across_models.

Naming:
  Predictions: quotes_<keyword_set>_<language>_<model>_<template>.csv
    keyword_set in {c,k}
    language in {en,en_nmt,it}
    model in {M1..M8}
    template in {T0,T1}
    columns: text,time,<kw1>,...,<kwN>

  Ground truth: quotes_<keyword_set>_<language>.csv
    columns: text,<kw1>,...,<kwN>

Outputs (always) in outputs/analysis/:
  Language-separated:
    text_aggregate_metrics_c.csv
    text_aggregate_metrics_k.csv
    text_aggregate_metrics_all.csv         (aggregated across c+k)

  Cross-language (canonical text from EN GT):
    text_aggregate_metrics_c_xlang.csv
    text_aggregate_metrics_k_xlang.csv
    text_aggregate_metrics_all_xlang.csv   (aggregated across c+k)

Optional (with --write_breakdowns):
  For each tag above:
    text_model_breakdown_{tag}.csv
    text_keyword_errors_{tag}.csv
    text_experiment_breakdown_{tag}.csv

Usage:
  python detail_res_analysis.py
  python detail_res_analysis.py --threshold 0.7
  python detail_res_analysis.py --write_breakdowns
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
    p = argparse.ArgumentParser(description="Aggregate entailment results into per-text metrics.")
    p.add_argument("--pred_dir", type=str, default="outputs/raw_results", help="Folder with prediction CSV files.")
    p.add_argument("--gt_dir", type=str, default="data", help="Folder with ground-truth CSV files.")
    p.add_argument("--out_dir", type=str, default="outputs/analysis", help="Destination folder for analysis CSVs.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for y_pred -> predicted label.")
    p.add_argument("--write_breakdowns", action="store_true", help="Also write breakdown CSVs.")
    p.add_argument(
        "--strict_keywords",
        action="store_true",
        help=(
            "If set, enforce same number of keyword columns between GT and predictions for each (set,lang). "
            "Keyword NAMES may differ; only counts/order matter."
        ),
    )
    return p.parse_args()


def _uniq(cols: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


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
        [row_id, text, time, kw_id, y_pred, keyword_set, language, model, template]
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
        id_vars=["row_id", "text", "time"],
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


def aggregate_tables_by_id(
    data: pd.DataFrame,
    id_cols: List[str],
    text_col: str,
    kw_col_for_counts: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    data must contain:
      id_cols + [text_col, language, model, template, time, abs_error, correct, gt_num_entailed,
                 kw_col_for_counts, tp, fp, fn]
    """

    exp_cols = _uniq(id_cols + [text_col, "language", "model", "template"])
    model_cols = _uniq(id_cols + [text_col, "model"])
    text_id_cols = _uniq(id_cols + [text_col])

    per_text_exp = (
        data.groupby(exp_cols, dropna=False)
        .agg(
            gt_num_entailed=("gt_num_entailed", "first"),
            num_keywords=(kw_col_for_counts, "nunique"),
            num_pairs=(kw_col_for_counts, "size"),
            mean_time=("time", "mean"),
            accuracy=("correct", "mean"),
            mean_error=("abs_error", "mean"),
            max_error=("abs_error", "max"),
        )
        .reset_index()
    )

    per_text_model = (
        data.groupby(model_cols, dropna=False)
        .agg(
            gt_num_entailed=("gt_num_entailed", "first"),
            accuracy=("correct", "mean"),
            mean_error=("abs_error", "mean"),
            num_pairs=(kw_col_for_counts, "size"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
        )
        .reset_index()
    )

    # Micro-F1 over keyword decisions for that (text-id, model)
    tp = per_text_model["tp"].astype(float)
    fp = per_text_model["fp"].astype(float)
    fn = per_text_model["fn"].astype(float)

    denom_p = tp + fp
    denom_r = tp + fn
    precision = np.where(denom_p > 0, tp / denom_p, 0.0)
    recall = np.where(denom_r > 0, tp / denom_r, 0.0)
    denom_f = precision + recall
    per_text_model["f_score"] = np.where(denom_f > 0, 2.0 * precision * recall / denom_f, 0.0)

    per_text = (
        data.groupby(text_id_cols, dropna=False)
        .agg(
            gt_num_entailed=("gt_num_entailed", "first"),
            languages_present=("language", lambda s: ",".join(sorted(set(s.dropna().astype(str))))),
            models_present=("model", lambda s: ",".join(sorted(set(s.dropna().astype(str))))),
            templates_present=("template", lambda s: ",".join(sorted(set(s.dropna().astype(str))))),
            num_keywords_total=(kw_col_for_counts, "nunique"),
            num_predictions_total=(kw_col_for_counts, "size"),
            mean_error_all=("abs_error", "mean"),
            median_error_all=("abs_error", "median"),
            worst_error_all=("abs_error", "max"),
            accuracy_all=("correct", "mean"),
            mean_time_all=("time", "mean"),
        )
        .reset_index()
    )

    model_rob = (
        per_text_model.groupby(text_id_cols, dropna=False)
        .agg(
            worst_accuracy_by_model=("accuracy", "min"),
            best_accuracy_by_model=("accuracy", "max"),
            std_accuracy_across_models=("accuracy", "std"),
            mean_accuracy_across_models=("accuracy", "mean"),
            mean_f_score_across_models=("f_score", "mean"),
        )
        .reset_index()
    )

    per_text = per_text.merge(model_rob, on=text_id_cols, how="left")
    per_text["std_accuracy_across_models"] = per_text["std_accuracy_across_models"].fillna(0.0)

    per_text["best_score"] = per_text["accuracy_all"] - per_text["std_accuracy_across_models"]
    per_text["worst_score"] = (
        per_text["mean_error_all"] + per_text["worst_error_all"] + per_text["std_accuracy_across_models"]
    )

    per_text = per_text.sort_values(["worst_score", "num_predictions_total"], ascending=[False, False])

    # Rename display text column to 'text' only at the end
    per_text = per_text.rename(columns={text_col: "text"})
    per_text_model = per_text_model.rename(columns={text_col: "text"})
    per_text_exp = per_text_exp.rename(columns={text_col: "text"})

    return per_text, per_text_model, per_text_exp


def text_keyword_table_by_id(
    df: pd.DataFrame,
    id_cols: List[str],
    text_col: str,
    kw_col: str,
    label_col: str | None = None,
) -> pd.DataFrame:
    group_cols = _uniq(id_cols + [text_col, kw_col])
    if label_col:
        group_cols = _uniq(group_cols + [label_col])

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            gt_num_entailed=("gt_num_entailed", "first"),
            mean_error=("abs_error", "mean"),
            max_error=("abs_error", "max"),
            accuracy=("correct", "mean"),
            num_pairs=(kw_col, "size"),
        )
        .reset_index()
        .rename(columns={text_col: "text"})
    )
    return out


def write_outputs(
    out_dir: Path,
    tag: str,
    per_text: pd.DataFrame,
    per_text_model: pd.DataFrame,
    per_text_exp: pd.DataFrame,
    write_breakdowns: bool,
    df_for_kw: pd.DataFrame,
    id_cols: List[str],
    text_col: str,
    kw_col_for_breakdown: str,
    kw_label_col: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_text.to_csv(out_dir / f"text_aggregate_metrics_{tag}.csv", index=False)

    if write_breakdowns:
        per_text_model.to_csv(out_dir / f"text_model_breakdown_{tag}.csv", index=False)
        text_keyword_table_by_id(df_for_kw, id_cols, text_col, kw_col_for_breakdown, kw_label_col).to_csv(
            out_dir / f"text_keyword_errors_{tag}.csv", index=False
        )
        per_text_exp.to_csv(out_dir / f"text_experiment_breakdown_{tag}.csv", index=False)


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

    for ks in ["c", "k"]:
        if (ks, "en") not in gt_index:
            raise FileNotFoundError(f"Missing canonical English ground truth: {gt_dir / f'quotes_{ks}_en.csv'}")

    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file() and PRED_RE.match(p.name)])
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}. Expected names like quotes_c_en_M1_T0.csv")

    # Build canonical EN keyword labels by position:
    # (keyword_set, kw_id) -> kw_label_en (from EN GT column names)
    kw_label_en: Dict[Tuple[str, int], str] = {}

    # Canonical EN quote text and entailed count per (set,row_id)
    en_text_map: Dict[Tuple[str, int], str] = {}
    en_entailed_map: Dict[Tuple[str, int], int] = {}

    # Totals across sets per row_id for *_all outputs
    en_text_all_map: Dict[int, str] = {}
    en_entailed_total_map: Dict[int, int] = {}

    # Load EN ground truth for each set and build label/count maps
    for ks in ["c", "k"]:
        gt_en_path = gt_index[(ks, "en")]
        gt_en = safe_read_csv(gt_en_path).reset_index(drop=False).rename(columns={"index": "row_id"})
        if "text" not in gt_en.columns:
            raise ValueError(f"{gt_en_path} missing required column 'text'")

        kw_cols = [c for c in gt_en.columns if c not in ("row_id", "text")]
        if not kw_cols:
            raise ValueError(f"{gt_en_path} has no keyword columns.")
        gt_en = to_numeric_cols(gt_en, kw_cols)

        for i, col in enumerate(kw_cols):
            kw_label_en[(ks, i)] = col

        for _, r in gt_en.iterrows():
            rid = int(r["row_id"])
            en_text_map[(ks, rid)] = str(r["text"])
            en_entailed_map[(ks, rid)] = int((r[kw_cols] == 1).sum())

    # Totals across sets by row_id (assumes same quote list/order across sets)
    all_row_ids = sorted({rid for (_, rid) in en_text_map.keys()})
    for rid in all_row_ids:
        en_text_all_map[rid] = en_text_map.get(("c", rid), en_text_map.get(("k", rid), ""))
        en_entailed_total_map[rid] = int(en_entailed_map.get(("c", rid), 0) + en_entailed_map.get(("k", rid), 0))

    # Cache GT longs by (keyword_set, language) and keyword counts
    gt_long_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    gt_kwcount_cache: Dict[Tuple[str, str], int] = {}

    skipped: List[str] = []
    all_rows: List[pd.DataFrame] = []

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

        if args.strict_keywords:
            if len(pred_kw_cols) != gt_kwcount_cache[key]:
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

        # Canonical EN texts + entailed counts
        merged["text_en_set"] = merged.apply(lambda r: en_text_map[(r["keyword_set"], int(r["row_id"]))], axis=1)
        merged["text_en_all"] = merged["row_id"].map(en_text_all_map)
        merged["gt_num_entailed_set"] = merged.apply(
            lambda r: en_entailed_map[(r["keyword_set"], int(r["row_id"]))], axis=1
        ).astype(int)
        merged["gt_num_entailed_total"] = merged["row_id"].map(en_entailed_total_map).astype(int)

        # Human-readable EN label for this keyword position
        merged["kw_label_en"] = merged.apply(lambda r: kw_label_en[(r["keyword_set"], int(r["kw_id"]))], axis=1)

        # Disambiguate across sets for *_all outputs
        merged["kw_global_id"] = merged["keyword_set"].astype(str) + "::" + merged["kw_id"].astype(str)
        merged["kw_global_label_en"] = merged["keyword_set"].astype(str) + "::" + merged["kw_label_en"].astype(str)

        # Derived metrics
        merged["abs_error"] = (merged["y_pred"] - merged["y_true"]).abs()
        merged["pred_label"] = (merged["y_pred"] >= thr).astype(int)

        # Positive class = entailment (y_true==1)
        merged["tp"] = ((merged["pred_label"] == 1) & (merged["y_true"] == 1)).astype(int)
        merged["fp"] = ((merged["pred_label"] == 1) & (merged["y_true"] == 0)).astype(int)
        merged["fn"] = ((merged["pred_label"] == 0) & (merged["y_true"] == 1)).astype(int)

        merged["correct"] = (merged["pred_label"] == merged["y_true"].astype(int)).astype(float)

        all_rows.append(merged)

    if not all_rows:
        raise RuntimeError("No usable data after merging predictions with ground truth.")

    data_all = pd.concat(all_rows, ignore_index=True)

    # -----------------------------
    # 1) LANGUAGE-SEPARATED OUTPUTS
    # -----------------------------

    # 1a) Split files: c / k
    id_cols_lang_split = ["keyword_set", "language", "row_id"]
    for ks in ["c", "k"]:
        subset = data_all[data_all["keyword_set"] == ks].copy()
        if subset.empty:
            continue
        subset["gt_num_entailed"] = subset["gt_num_entailed_set"]

        per_text, per_text_model, per_text_exp = aggregate_tables_by_id(
            subset,
            id_cols=id_cols_lang_split,
            text_col="text",
            kw_col_for_counts="kw_id",
        )
        write_outputs(
            out_dir=out_dir,
            tag=ks,
            per_text=per_text,
            per_text_model=per_text_model,
            per_text_exp=per_text_exp,
            write_breakdowns=args.write_breakdowns,
            df_for_kw=subset,
            id_cols=id_cols_lang_split,
            text_col="text",
            kw_col_for_breakdown="kw_id",
            kw_label_col="kw_label_en",
        )

    # 1b) ALL language-separated: aggregate across keyword sets
    id_cols_lang_all = ["language", "row_id"]
    all_lang = data_all.copy()
    all_lang["gt_num_entailed"] = all_lang["gt_num_entailed_total"]

    per_text_all, per_text_model_all, per_text_exp_all = aggregate_tables_by_id(
        all_lang,
        id_cols=id_cols_lang_all,
        text_col="text",
        kw_col_for_counts="kw_global_id",
    )
    write_outputs(
        out_dir=out_dir,
        tag="all",
        per_text=per_text_all,
        per_text_model=per_text_model_all,
        per_text_exp=per_text_exp_all,
        write_breakdowns=args.write_breakdowns,
        df_for_kw=all_lang,
        id_cols=id_cols_lang_all,
        text_col="text",
        kw_col_for_breakdown="kw_global_id",
        kw_label_col="kw_global_label_en",
    )

    # --------------------------------
    # 2) CROSS-LANGUAGE (ROW-ALIGNED) OUTPUTS
    # --------------------------------

    # 2a) Split: c / k cross-language
    id_cols_x_split = ["keyword_set", "row_id"]
    for ks in ["c", "k"]:
        subset = data_all[data_all["keyword_set"] == ks].copy()
        if subset.empty:
            continue
        subset["gt_num_entailed"] = subset["gt_num_entailed_set"]

        per_text, per_text_model, per_text_exp = aggregate_tables_by_id(
            subset,
            id_cols=id_cols_x_split,
            text_col="text_en_set",
            kw_col_for_counts="kw_id",
        )
        write_outputs(
            out_dir=out_dir,
            tag=f"{ks}_xlang",
            per_text=per_text,
            per_text_model=per_text_model,
            per_text_exp=per_text_exp,
            write_breakdowns=args.write_breakdowns,
            df_for_kw=subset,
            id_cols=id_cols_x_split,
            text_col="text_en_set",
            kw_col_for_breakdown="kw_id",
            kw_label_col="kw_label_en",
        )

    # 2b) ALL cross-language: aggregate across keyword sets
    id_cols_x_all = ["row_id"]
    all_x = data_all.copy()
    all_x["gt_num_entailed"] = all_x["gt_num_entailed_total"]

    per_text_all_x, per_text_model_all_x, per_text_exp_all_x = aggregate_tables_by_id(
        all_x,
        id_cols=id_cols_x_all,
        text_col="text_en_all",
        kw_col_for_counts="kw_global_id",
    )
    write_outputs(
        out_dir=out_dir,
        tag="all_xlang",
        per_text=per_text_all_x,
        per_text_model=per_text_model_all_x,
        per_text_exp=per_text_exp_all_x,
        write_breakdowns=args.write_breakdowns,
        df_for_kw=all_x,
        id_cols=id_cols_x_all,
        text_col="text_en_all",
        kw_col_for_breakdown="kw_global_id",
        kw_label_col="kw_global_label_en",
    )

    if skipped:
        (out_dir / "aggregation_skipped_files.txt").write_text("\n".join(skipped), encoding="utf-8")


if __name__ == "__main__":
    main()
