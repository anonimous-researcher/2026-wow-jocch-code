import pathlib
import pandas
import matplotlib.pyplot as plt
import seaborn
import re
import argparse

seaborn.set(style="whitegrid")

RAW_RESULT_REGEX = re.compile(r"quotes_([kc])_(it|en|en_nmt)_(M[1-8])_(T[01])\.csv")
MODEL_PALETTE = {
    "M1": "navy",
    "M2": "royalblue",
    "M3": "cornflowerblue",
    "M4": "skyblue",
    "M5": "darkred",
    "M6": "firebrick",
    "M7": "indianred",
    "M8": "salmon"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cactus plot from raw result times.")
    parser.add_argument("--input_folder", required=True, type=str, help="Folder containing quotes_*.csv files.")
    parser.add_argument("--output_folder", required=True, type=str, help="Folder where cactus plot will be saved.")
    return parser.parse_args()


def build_cactus_plot(input_folder: pathlib.Path, output_file: pathlib.Path):
    model_times = {}

    for file in input_folder.glob("quotes_*.csv"):
        match = RAW_RESULT_REGEX.match(file.name)
        if not match:
            continue

        _, _, model, _ = match.groups()
        df = pandas.read_csv(file)

        if model not in model_times:
            model_times[model] = []

        model_times[model].extend(df["time"].dropna().tolist())

    for model in model_times:
        model_times[model].sort()

    plt.figure(figsize=(10, 6))
    for model in sorted(model_times.keys()):
        times = model_times[model]
        plt.plot(
            range(1, len(times) + 1),
            times,
            label=model,
            color=MODEL_PALETTE.get(model),
            linewidth=2
        )

    plt.xlabel("Number of Samples (sorted by time)", fontsize="large", weight="bold")
    plt.ylabel("Computation Time (s)", fontsize="large", weight="bold")
    plt.title("Cactus Plot of Inference Times per Model", fontsize="large", weight="bold")
    plt.legend(title="Model")
    plt.tight_layout()

    plt.savefig(output_file)
    print(f"📈 Saved cactus plot to {output_file}")


def main():
    args = parse_args()
    input_folder = pathlib.Path(args.input_folder)
    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "cactus_plot_times.pdf"
    build_cactus_plot(input_folder, output_file)


if __name__ == "__main__":
    main()
