
import argparse
import pandas
import re
import logging
import pathlib


FILENAME_REGEX = re.compile(r"quotes_([kc])_(it|en|en_nmt)_(M[1-8])_(T[01])\.csv")
GROUND_TRUTH_REGEX = re.compile(r"quotes_([kc])_(it|en|en_nmt)\.csv")


def get_classification_df(probabilities: pandas.DataFrame, threshold: float) -> pandas.DataFrame:
    """
    Convert a probability DataFrame to a binary classification DataFrame using the given threshold.
    Columns 'text' and 'time' are preserved.
    """
    classification = probabilities.copy()
    class_columns = [col for col in classification.columns if col not in ["text", "time"]]
    classification[class_columns] = (classification[class_columns] > threshold).astype(int)
    return classification


def compute_metrics(predicted: pandas.DataFrame, ground_truth: pandas.DataFrame) -> pandas.DataFrame:
    """
    Compute classification metrics (Precision, Recall, F-Score, Accuracy) per label.
    Assumes 'text' is present in both DataFrames and 'time' is in `predicted`.
    """
    label_columns = [col for col in ground_truth.columns if col != "text"]
    metrics = {}

    for label in label_columns:
        y_true = ground_truth[label]
        y_pred = predicted[label]

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        total = len(y_true)
        total_pos = y_true.sum()
        total_neg = total - total_pos

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        accuracy = (tp + tn) / total if total else 0
        pos_percentage = total_pos / total if total else 0

        metrics[label] = {
            "True Positives": tp,
            "False Positives": fp,
            "True Negatives": tn,
            "False Negatives": fn,
            "Total Positives": total_pos,
            "Total Negatives": total_neg,
            "Positive Percentage": pos_percentage,
            "Precision": precision,
            "Recall": recall,
            "F-Score": f_score,
            "Accuracy": accuracy
        }

    return pandas.DataFrame(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for classification results.")
    parser.add_argument("--results_folder", required=True, type=str,
                        help="Folder containing raw result .csv files")
    parser.add_argument("--ground_truth_folder", required=True, type=str,
                        help="Folder containing ground truth .csv files")
    parser.add_argument("--output_folder", required=True, type=str,
                        help="Folder to store output metric files")
    parser.add_argument("--threshold", type=float, default=0.875,
                        help="Classification threshold (default: 0.625)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level: DEBUG, INFO, WARNING, ERROR")
    return parser.parse_args()


def setup_logger(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), "INFO"),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def evaluate_all(results_dir: pathlib.Path, ground_truth_dir: pathlib.Path, output_dir: pathlib.Path, threshold: float):
    logger = logging.getLogger("MetricsEvaluator")

    # Load ground truth files
    ground_truth_map = {}
    for gt_file in ground_truth_dir.glob("quotes_*.csv"):
        match = GROUND_TRUTH_REGEX.match(gt_file.name)
        if match:
            key_type, language = match.groups()
            df = pandas.read_csv(gt_file)
            ground_truth_map[(key_type, language)] = df
            logger.debug(f"Loaded ground truth: {gt_file.name}")

    # Evaluate each results file
    for result_file in results_dir.glob("quotes_*.csv"):
        match = FILENAME_REGEX.match(result_file.name)
        if not match:
            logger.debug(f"Ignoring non-matching file: {result_file.name}")
            continue

        key_type, language, model, template = match.groups()
        key = (key_type, language)

        if key not in ground_truth_map:
            logger.warning(f"No matching ground truth for {result_file.name}, skipping.")
            continue

        logger.info(f"Processing {result_file.name} with threshold {threshold}")

        try:
            result_df = pandas.read_csv(result_file)
            ground_truth_df = ground_truth_map[key]

            # Ensure same samples by 'text'
            result_df = result_df[result_df["text"].isin(ground_truth_df["text"])]
            ground_truth_df = ground_truth_df[ground_truth_df["text"].isin(result_df["text"])]
            result_df = result_df.sort_values("text").reset_index(drop=True)
            ground_truth_df = ground_truth_df.sort_values("text").reset_index(drop=True)

            classification_df = get_classification_df(result_df, threshold=threshold)
            metrics_df = compute_metrics(classification_df, ground_truth_df)

            threshold_pct = threshold
            output_filename = f"metrics_{key_type}_{language}_{model}_{template}_{threshold_pct}.csv"

            output_path = output_dir / output_filename
            metrics_df.T.to_csv(output_path, index=True)

            logger.info(f"Saved metrics to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to process {result_file.name}: {e}")


def main():
    args = parse_args()
    setup_logger(args.log_level)

    results_dir = pathlib.Path(args.results_folder)
    ground_truth_dir = pathlib.Path(args.ground_truth_folder)
    output_dir = pathlib.Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_all(results_dir, ground_truth_dir, output_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()
