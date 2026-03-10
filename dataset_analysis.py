import argparse
import pathlib
import pandas
import matplotlib.pyplot as plt
import seaborn
import re


seaborn.set(style="whitegrid")
FILENAME_REGEX = re.compile(r"quotes_([kc])_(it|en|en_nmt)\.csv")


def plot_text_length_histogram(lengths: pandas.Series, output_file: pathlib.Path):
    label_fontsize = 20
    tick_fontsize = 15

    plt.figure(figsize=(10, 6))
    seaborn.histplot(lengths, bins=30, kde=True, color="slateblue")

    plt.xlabel("Length", fontsize=label_fontsize, weight="bold")
    plt.ylabel("# of Quotes", fontsize=label_fontsize, weight="bold")

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"📊 Saved text length histogram to {output_file}")


def plot_keyword_frequency_histogram(counts: list[int], output_file: pathlib.Path):
    label_fontsize = 20
    tick_fontsize = 15

    plt.figure(figsize=(10, 6))

    if "_k_" in str(output_file):
        label = "Keywords"
        color = "seagreen"
    else:
        label = "Concepts"
        color = "goldenrod"

    seaborn.histplot(counts, bins=20, kde=True, color=color)

    plt.xlabel("# of Occurrences", fontsize=label_fontsize, weight="bold")
    plt.ylabel(f"# of {label}", fontsize=label_fontsize, weight="bold")

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"📊 Saved keyword frequency histogram to {output_file}")


def extract_key_and_language(filename: str) -> tuple[str, str]:
    match = FILENAME_REGEX.match(filename)
    if not match:
        raise ValueError(f"Filename '{filename}' must match format 'quotes_<key_type>_<language>.csv'")
    return match.groups()  # returns (key_type, language)


def analyze_dataset(input_file: pathlib.Path, output_folder: pathlib.Path, key_type: str, language: str):
    df = pandas.read_csv(input_file)

    if "text" not in df.columns:
        raise ValueError(f"The file '{input_file.name}' does not contain a 'text' column.")

    output_folder.mkdir(parents=True, exist_ok=True)

    df["text_length"] = df["text"].astype(str).apply(len)
    min_len = df["text_length"].min()
    max_len = df["text_length"].max()

    # Plot histogram of text lengths
    text_len_plot = output_folder / f"hist_text_lengths_{key_type}_{language}.pdf"
    plot_text_length_histogram(df["text_length"], text_len_plot)

    # Count 1s for each label column
    keyword_columns = [col for col in df.columns if col not in ["text", "text_length"]]
    keyword_counts = {col: (df[col] == 1).sum() for col in keyword_columns}
    sorted_keyword_counts = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    count_values = [count for _, count in sorted_keyword_counts]
    min_count = min(count_values)
    max_count = max(count_values)

    # Plot histogram of keyword frequencies
    keyword_freq_plot = output_folder / f"hist_keyword_frequencies_{key_type}_{language}.pdf"
    plot_keyword_frequency_histogram(count_values, keyword_freq_plot)

    # Write summary to txt file
    summary_file = output_folder / f"summary_{key_type}_{language}.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write(f"Text length:\n")
        f.write(f"  Min: {min_len}\n")
        f.write(f"  Max: {max_len}\n\n")
        f.write(f"Keyword frequency (value = 1):\n")
        f.write(f"  Min: {min_count}\n")
        f.write(f"  Max: {max_count}\n")
        f.write(f"  Total keywords: {len(keyword_columns)}\n\n")
        f.write(f"Per-keyword frequency counts (sorted):\n")
        for keyword, count in sorted_keyword_counts:
            f.write(f"  {keyword}: {count}\n")

    print(f"📝 Saved summary to {summary_file}")

    return {
        "dataset": input_file.name,
        "key_type": key_type,
        "language": language,
        "text_length_min": min_len,
        "text_length_max": max_len,
        "text_length_mean": df["text_length"].mean(),
        "keyword_1_min": min_count,
        "keyword_1_max": max_count,
        "keyword_1_mean": sum(count_values) / len(count_values),
        "keyword_count": len(count_values)
    }


def generate_global_summary(summary_data: list[dict], output_folder: pathlib.Path):
    df_summary = pandas.DataFrame(summary_data)
    summary_csv = output_folder / "global_dataset_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"🧾 Saved global dataset summary to {summary_csv}")

    # Plot average text length
    plt.figure(figsize=(10, 6))
    seaborn.barplot(
        data=df_summary.sort_values("text_length_mean"),
        x="dataset",
        y="text_length_mean",
        palette="Blues_d"
    )
    plt.title("Average Text Length per Dataset", fontsize="large", weight="bold")
    plt.xlabel("Dataset", fontsize="medium", weight="bold")
    plt.ylabel("Mean Text Length", fontsize="medium", weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_folder / "avg_text_length_comparison.pdf")
    plt.close()
    print(f"📊 Saved average text length comparison plot")

    # Plot average keyword frequency
    plt.figure(figsize=(10, 6))
    seaborn.barplot(
        data=df_summary.sort_values("keyword_1_mean"),
        x="dataset",
        y="keyword_1_mean",
        palette="Greens_d"
    )
    plt.title("Average Keyword Frequency per Dataset", fontsize="large", weight="bold")
    plt.xlabel("Dataset", fontsize="medium", weight="bold")
    plt.ylabel("Mean Frequency (value=1)", fontsize="medium", weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_folder / "avg_keyword_freq_comparison.pdf")
    plt.close()
    print(f"📊 Saved average keyword frequency comparison plot")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze labeled datasets in a folder.")
    parser.add_argument("--input_folder", type=str, default="data/",
                        help="Folder containing CSV datasets (default: data/).")
    parser.add_argument("--output_folder", type=str, default="outputs/graphs/",
                        help="Folder where plots and summaries will be saved (default: 'outputs/graphs/').")
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = pathlib.Path(args.input_folder)
    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    summary_data = []

    for file in input_folder.glob("quotes_*.csv"):
        if FILENAME_REGEX.match(file.name):
            key_type, language = extract_key_and_language(file.name)
            print(f"🔍 Analyzing {file.name} ...")
            try:
                stats = analyze_dataset(file, output_folder, key_type, language)
                summary_data.append(stats)
            except Exception as e:
                print(f"❌ Skipping {file.name}: {e}")
        else:
            print(f"⏩ Ignored non-matching file: {file.name}")

    if summary_data:
        generate_global_summary(summary_data, output_folder)


if __name__ == "__main__":
    main()
