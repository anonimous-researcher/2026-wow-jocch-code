import argparse
import logging
import pathlib
import re

import pandas


FILENAME_REGEX = re.compile(
    r"quotes_([kc])_(it|en|en_nmt)_(M[1-8])_(T[01])\.csv"
)

GROUND_TRUTH_REGEX = re.compile(
    r"quotes_([kc])_(it|en|en_nmt)\.csv"
)


def compute_top_k_metrics(
    probabilities: pandas.DataFrame,
    ground_truth: pandas.DataFrame,
    k_values: list[int]
) -> pandas.DataFrame:
    """
    Compute per-excerpt Recall@K and Precision@K for multiple K values.

    For each excerpt, candidate labels are ranked by their raw probability
    in descending order. The first K labels are compared against the
    corresponding gold-standard labels.

    Recall@K = |G_i ∩ S_i^(K)| / |G_i|
    Precision@K = |G_i ∩ S_i^(K)| / K

    For excerpts with no gold-standard labels, Recall@K is undefined
    and is reported as NaN. Precision@K is reported as 0.

    Columns 'text' and 'time' are preserved from the probability DataFrame.
    """

    if not k_values:
        raise ValueError("At least one K value must be provided.")

    if any(k <= 0 for k in k_values):
        raise ValueError("All K values must be positive integers.")

    k_values = sorted(set(k_values))

    # All columns except metadata are candidate labels.
    label_columns = [
        col
        for col in probabilities.columns
        if col not in ["text", "time"]
    ]

    max_k = max(k_values)

    if max_k > len(label_columns):
        raise ValueError(
            f"Maximum K={max_k} is larger than the number of "
            f"candidate labels ({len(label_columns)})."
        )

    results = []

    for index in range(len(probabilities)):
        probability_row = probabilities.iloc[index]
        ground_truth_row = ground_truth.iloc[index]

        # Rank candidate labels by probability, descending.
        ranked_labels = (
            probability_row[label_columns]
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        # Gold-standard labels.
        gold_labels = {
            label
            for label in label_columns
            if ground_truth_row[label] == 1
        }

        result = {
            "text": probability_row["text"]
        }

        if "time" in probabilities.columns:
            result["time"] = probability_row["time"]

        # Compute metrics for each requested K.
        for k in k_values:
            top_k_labels = set(ranked_labels[:k])

            relevant_labels = top_k_labels.intersection(gold_labels)

            if gold_labels:
                recall_at_k = len(relevant_labels) / len(gold_labels)
            else:
                recall_at_k = float("nan")

            precision_at_k = len(relevant_labels) / k

            result[f"Recall@{k}"] = recall_at_k
            result[f"Precision@{k}"] = precision_at_k

        results.append(result)

    return pandas.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-excerpt Recall@K and Precision@K "
            "for ranking results."
        )
    )

    parser.add_argument(
        "--results_folder",
        type=str,
        default="outputs/raw_results/",
        help="Folder containing raw result .csv files"
    )

    parser.add_argument(
        "--ground_truth_folder",
        type=str,
        default="data/",
        help="Folder containing ground truth .csv files"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs/metrics_top_k/",
        help="Folder to store output metric files"
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[50],
        help=(
            "One or more K values for which Recall@K and Precision@K "
            "are computed (default: 3, 5, 10)"
        )
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR"
    )

    return parser.parse_args()


def setup_logger(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def evaluate_all(
    results_dir: pathlib.Path,
    ground_truth_dir: pathlib.Path,
    output_dir: pathlib.Path,
    k_values: list[int]
):
    logger = logging.getLogger("TopKMetricsEvaluator")

    k_values = sorted(set(k_values))

    # Evaluate each raw results file.
    for result_file in results_dir.glob("quotes_*.csv"):

        match = FILENAME_REGEX.fullmatch(result_file.name)

        if not match:
            logger.debug(
                f"Ignoring non-matching file: {result_file.name}"
            )
            continue

        key_type, language, model, template = match.groups()

        # Construct the ground-truth filename explicitly.
        expected_gt_filename = (
            f"quotes_{key_type}_{language}.csv"
        )

        ground_truth_file = ground_truth_dir / expected_gt_filename

        if not ground_truth_file.exists():
            logger.warning(
                f"No matching ground truth for {result_file.name}. "
                f"Expected: {expected_gt_filename}"
            )
            continue

        logger.info(
            f"Processing {result_file.name}"
        )
        logger.info(
            f"Using ground truth: {ground_truth_file.name}"
        )
        logger.info(
            f"K values: {k_values}"
        )

        try:
            result_df = pandas.read_csv(result_file)
            ground_truth_df = pandas.read_csv(ground_truth_file)

            # Verify that result and ground-truth files contain the same excerpts
            # in the same order.
            if len(result_df) != len(ground_truth_df):
                raise ValueError(
                    f"Different number of excerpts: "
                    f"{len(result_df)} in results vs "
                    f"{len(ground_truth_df)} in ground truth."
                )

            if not result_df["text"].equals(ground_truth_df["text"]):
                raise ValueError(
                    "Result and ground-truth excerpts are not in the same order."
                )

            # Candidate labels are all columns except metadata.
            label_columns = [
                col
                for col in result_df.columns
                if col not in ["text", "time"]
            ]

            # Verify that all candidate labels exist in the ground truth.
            missing_labels = [
                col
                for col in label_columns
                if col not in ground_truth_df.columns
            ]

            if missing_labels:
                raise ValueError(
                    "The following candidate labels are missing from "
                    f"the ground truth: {missing_labels}"
                )

            # Compute per-excerpt ranking metrics.
            metrics_df = compute_top_k_metrics(
                probabilities=result_df,
                ground_truth=ground_truth_df,
                k_values=k_values
            )

            # Encode all requested K values in the filename.
            k_string = "-".join(str(k) for k in k_values)

            output_filename = (
                f"metrics_top_k_{key_type}_{language}_"
                f"{model}_{template}_K{k_string}.csv"
            )

            output_path = output_dir / output_filename

            metrics_df.to_csv(
                output_path,
                index=False
            )

            logger.info(
                f"Saved metrics to: {output_path}"
            )

        except Exception as e:
            logger.error(
                f"Failed to process {result_file.name}: {e}"
            )

def aggregate_metrics(
    output_dir: pathlib.Path,
    k_values: list[int]
):
    """
    Aggregate all per-excerpt top-K metric files into a single CSV.

    For each metrics file, compute the mean of each Recall@K and
    Precision@K column across excerpts.

    The metadata encoded in the filename is unpacked into separate
    columns:
        - key_type: c or k
        - language: it, en, or en_nmt
        - model: M1-M8
        - template: T0 or T1

    NaN values in Recall@K are ignored when computing the mean.
    """

    logger = logging.getLogger("TopKMetricsEvaluator")

    k_string = "-".join(str(k) for k in k_values)

    metrics_files = [
        path
        for path in output_dir.glob(
            f"metrics_top_k_*_K{k_string}.csv"
        )
        if "summary" not in path.name
    ]

    if not metrics_files:
        logger.warning(
            "No per-excerpt metric files found for aggregation."
        )
        return

    aggregated_results = []

    metric_columns = [
        metric
        for k in k_values
        for metric in [
            f"Recall@{k}",
            f"Precision@{k}"
        ]
    ]

    for metrics_file in sorted(metrics_files):

        filename_match = re.fullmatch(
            rf"metrics_top_k_([kc])_(it|en|en_nmt)_(M[1-8])_(T[01])_K{k_string}\.csv",
            metrics_file.name
        )

        if not filename_match:
            logger.warning(
                f"Skipping file with unexpected filename: "
                f"{metrics_file.name}"
            )
            continue

        key_type, language, model, template = filename_match.groups()

        try:
            df = pandas.read_csv(metrics_file)

            row = {
                "key_type": key_type,
                "language": language,
                "model": model,
                "template": template
            }

            for column in metric_columns:
                if column not in df.columns:
                    raise ValueError(
                        f"Column '{column}' not found in "
                        f"{metrics_file.name}"
                    )

                # pandas mean() ignores NaN values by default.
                row[column] = df[column].mean()

            aggregated_results.append(row)

        except Exception as e:
            logger.error(
                f"Failed to aggregate {metrics_file.name}: {e}"
            )

    if not aggregated_results:
        logger.warning("No files could be aggregated.")
        return

    aggregated_df = pandas.DataFrame(
        aggregated_results,
        columns=[
            "key_type",
            "language",
            "model",
            "template"
        ] + metric_columns
    )

    output_filename = f"metrics_top_k_summary_K{k_string}.csv"
    output_path = output_dir / output_filename

    aggregated_df.to_csv(output_path, index=False)

    logger.info(
        f"Saved aggregated metrics to: {output_path}"
    )


def main():
    args = parse_args()

    setup_logger(args.log_level)

    results_dir = pathlib.Path(args.results_folder)
    ground_truth_dir = pathlib.Path(args.ground_truth_folder)
    output_dir = pathlib.Path(args.output_folder)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluate_all(
        results_dir=results_dir,
        ground_truth_dir=ground_truth_dir,
        output_dir=output_dir,
        k_values=args.k
    )

    aggregate_metrics(
        output_dir=output_dir,
        k_values=args.k
    )


if __name__ == "__main__":
    main()