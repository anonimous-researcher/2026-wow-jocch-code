import pandas
import matplotlib.pyplot as plt
import seaborn
import pathlib
import re
import argparse
import numpy

seaborn.set(style="whitegrid")

# Regex to extract metadata from file names
METRICS_FILE_REGEX = re.compile(r"metrics_([kc])_(it|en|en_nmt)_(M[1-8])_(T[01])_([\d]+(?:\.\d+)?)\.csv")

# Color palette for models
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

LANGUAGE_COLOR_PALETTE = {
    "en": "#66c2a5",     # you can adjust colors as you prefer
    "en_nmt": "#fc8d62",
    "it": "#8da0cb"
}

# Fixed order of languages
LANGUAGE_ORDER = ["it", "en", "en_nmt"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from evaluation metrics.")
    parser.add_argument("--metrics_folder", required=True, type=str,
                        help="Folder containing metrics_*.csv files.")
    parser.add_argument("--output_folder", required=True, type=str,
                        help="Folder to store generated plots.")
    return parser.parse_args()


def plot_boxplot_distribution(df: pandas.DataFrame, metric: str, output_file: pathlib.Path):
    # Work with all raw (non-aggregated) rows
    df_filtered = df.copy()

    # Ensure consistent model order
    model_order = sorted(df_filtered["model"].unique())
    df_filtered["model"] = pandas.Categorical(df_filtered["model"], categories=model_order, ordered=True)

    # Set consistent template and key_type order
    df_filtered["template"] = pandas.Categorical(df_filtered["template"], categories=["T0", "T1"], ordered=True)
    df_filtered["key_type"] = pandas.Categorical(df_filtered["key_type"], categories=["k", "c"], ordered=True)

    # Create the boxplot grid
    plot = seaborn.catplot(
        data=df_filtered,
        x="model",
        y=metric,
        hue="language",
        kind="box",
        col="key_type",
        row="template",
        palette=LANGUAGE_COLOR_PALETTE,
        hue_order=LANGUAGE_ORDER,
        height=4,
        aspect=1.5,
        dodge=True,
        linewidth=0.8,
        sharey=True
    )

    # Remove x-axis labels from individual plots
    for ax in plot.axes[-1]:
        ax.set_xlabel("")

    # Set y-axis label only on the leftmost column
    for row_axes in plot.axes:
        for i, ax in enumerate(row_axes):
            ax.set_ylabel(metric if i == 0 else "")

    # Remove default facet titles
    for ax in plot.axes.flat:
        ax.set_title("")

    # Add custom column headers only to the top row
    col_titles = {"k": "Keywords", "c": "Concepts"}
    for i, key_type in enumerate(plot.col_names):
        ax = plot.axes[0, i]
        ax.set_title(col_titles.get(key_type, key_type), fontsize="large", weight="bold")

    # Add row labels manually to the right
    row_titles = {"T0": "T0", "T1": "T1"}
    for i, template in enumerate(plot.row_names):
        ax = plot.axes[i, -1]
        ax.annotate(
            row_titles.get(template, template),
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            rotation=270,
            ha="center",
            va="center",
            fontsize="large",
            weight="bold"
        )

    # Add single shared x-axis label for "Model"
    plot.fig.text(
        0.485, 0.01, "Models",
        ha="center",
        va="center",
        fontsize="large",
        weight="bold"
    )

    # Save to PDF
    plot.savefig(output_file)
    print(f"📦 Saved boxplot to {output_file}")


def plot_template_effect(df: pandas.DataFrame, metric: str, output_file: pathlib.Path):
    # Compute the average F-Score (or other metric) manually
    line_df = (
        df.groupby(["model", "language", "key_type", "template"], as_index=False)
          .agg({metric: "mean"})
    )

    # Sort models
    models = sorted(line_df["model"].unique())

    # Define markers and linestyles
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    linestyles = ["-", "--", ":", "-."]
    style_dict = {
        model: {
            "marker": markers[i % len(markers)],
            "linestyle": linestyles[i % len(linestyles)]
        }
        for i, model in enumerate(models)
    }

    # Ordering for facets and labels
    language_order = ["en_nmt", "en", "it"]
    key_type_order = ["k", "c"]
    col_titles = {"en_nmt": "en_nmt", "en": "en", "it": "it"}
    row_titles = {"k": "Keywords", "c": "Concepts"}

    line_df["language"] = pandas.Categorical(line_df["language"], categories=language_order, ordered=True)
    line_df["key_type"] = pandas.Categorical(line_df["key_type"], categories=key_type_order, ordered=True)
    line_df["template"] = pandas.Categorical(line_df["template"], categories=["T0", "T1"], ordered=True)

    # Build plot
    plot = seaborn.catplot(
        data=line_df,
        x="template",
        y=metric,
        hue="model",
        hue_order=models,
        col="language",
        row="key_type",
        kind="point",
        palette=MODEL_PALETTE,
        height=4,
        aspect=1.2,
        markers=[style_dict[m]["marker"] for m in models],
        linestyles=[style_dict[m]["linestyle"] for m in models],
        legend=False,
        errorbar=None  # no confidence intervals
    )

    # Remove default facet titles
    plot.set_titles("")

    # Custom column titles (top row only)
    for i, language in enumerate(language_order):
        ax = plot.axes[0, i]
        ax.set_title(col_titles.get(language, language), fontsize="large", weight="bold")

    # Custom row titles on the right
    for j, key_type in enumerate(key_type_order):
        ax = plot.axes[j, -1]
        ax.annotate(
            row_titles.get(key_type, key_type),
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            rotation=270,
            ha="center",
            va="center",
            fontsize="large",
            weight="bold"
        )

    # Clean x-labels and add global "Label Template" label
    for ax in plot.axes[-1]:
        ax.set_xlabel("")
    plot.fig.text(0.465, 0.04, "Label Template", ha="center", va="center", fontsize="large", weight="bold")

    # Y-axis label only on first column
    for row_axes in plot.axes:
        for i, ax in enumerate(row_axes):
            ax.set_ylabel(metric if i == 0 else "")

    # Manual legend on the right
    handles, labels = plot._legend_data.values(), plot._legend_data.keys()
    plot.fig.legend(
        handles=handles,
        labels=labels,
        title="Model",
        loc="center right",
        bbox_to_anchor=(0.95, 0.53),
        frameon=False
    )

    # Final layout and save
    plot.fig.subplots_adjust(top=0.95, bottom=0.12, right=0.87)
    plot.savefig(output_file)
    print(f"📦 Saved template effect plot to {output_file}")


def plot_heatmap_summary(df: pandas.DataFrame, metric: str, output_file: pathlib.Path):
    # Compute mean metric from raw (non-aggregated) data
    heatmap_data = (
        df
        .groupby(["model", "language"], as_index=True)[metric]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    heatmap = seaborn.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Mean {metric}"},
        ax=ax
    )

    # Bold and padded axis labels
    ax.set_ylabel("Model", labelpad=10, fontsize="medium", weight="bold")
    ax.set_xlabel("Language", labelpad=10, fontsize="medium", weight="bold")

    # Bold colorbar label with proper spacing
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size("medium")
    colorbar.ax.yaxis.label.set_weight("bold")
    colorbar.ax.yaxis.labelpad = 10

    plt.savefig(output_file)
    print(f"📦 Saved heatmap to {output_file}")


def plot_heatmap_summary_by_keytype_template(df: pandas.DataFrame, metric: str, output_file: pathlib.Path):

    seaborn.set(style="whitegrid")

    key_types = ["k", "c"]
    templates = ["T0", "T1"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    vmin, vmax = 0.05, 0.35  # Set consistent scale across heatmaps
    cmap = "Greens"

    for i, (key_type, template) in enumerate([(k, t) for k in key_types for t in templates]):
        ax = axes[i]
        subset = df[(df["key_type"] == key_type) & (df["template"] == template)]

        heatmap_data = (
            subset
            .groupby(["model", "language"], as_index=True)[metric]
            .mean()
            .unstack()
        )

        seaborn.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            ax=ax
        )

        if key_type == "k":
            ax.set_title(f"Keywords - {template}", fontsize="large", weight="bold")
        else:
            ax.set_title(f"Concepts - {template}", fontsize="large", weight="bold")

        ax.set_xlabel("Language", fontsize="medium", weight="bold", labelpad=8)
        ax.set_ylabel("Model", fontsize="medium", weight="bold", labelpad=8)

    # Add a single colorbar aligned to the full figure
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.75, pad=0.02)
    cbar.set_label(f"Mean {metric}", fontsize="medium", weight="bold", labelpad=10)

    plt.savefig(output_file)
    print(f"📊 Saved 2x2 heatmap grid to {output_file}")
    plt.close()


def plot_heatmap_by_threshold(df: pandas.DataFrame, metric: str, output_file: pathlib.Path):
    thresholds = sorted(df["threshold"].unique())[:4]  # Limit to 4 thresholds
    n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    if metric == "F-Score":
        vmin = 0
        vmax = 0.25
        cmap = "Purples"
    elif metric == "Precision":
        vmin = 0
        vmax = 0.25
        cmap = "Blues"
    else:
        vmin = 0.3
        vmax = 0.9
        cmap = "Reds"

    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        subset = df[df["threshold"] == threshold]

        heatmap_data = (
            subset
            .groupby(["model", "language"], as_index=True)[metric]
            .mean()
            .unstack()
        )

        seaborn.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax
        )

        ax.set_title(f"Threshold = {threshold}", fontsize="large", weight="bold")
        ax.set_xlabel("Language", labelpad=15, fontsize="medium", weight="bold")
        ax.set_ylabel("Model", labelpad=15, fontsize="medium", weight="bold")

    # Hide any unused subplots (not needed if exactly 4, but safe for future)
    for j in range(len(thresholds), len(axes)):
        fig.delaxes(axes[j])

    # Add a single colorbar aligned to the full figure
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar on the right
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.7, pad=0.02)
    cbar.set_label(f"Mean {metric}", fontsize="medium", weight="bold", labelpad=10)

    plt.savefig(output_file)
    print(f"📦 Saved threshold heatmap grid to {output_file}")


ALL_METRICS = ["Precision", "Recall", "F-Score"]

def main():
    args = parse_args()

    metrics_path = pathlib.Path(args.metrics_folder)
    output_path = pathlib.Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all relevant metric CSV files
    records = []
    for file in metrics_path.glob("metrics_*.csv"):
        match = METRICS_FILE_REGEX.match(file.name)
        if not match:
            continue

        key_type, language, model, template, threshold_str = match.groups()
        threshold = float(threshold_str)

        df = pandas.read_csv(file, index_col=0).reset_index()
        df.rename(columns={"index": "keyword"}, inplace=True)
        df["key_type"] = key_type
        df["language"] = language
        df["model"] = model
        df["template"] = template
        df["threshold"] = threshold

        records.append(df)

    # Combine all metric rows
    combined_df = pandas.concat(records, ignore_index=True)

    # Loop through each metric
    for metric_col in ALL_METRICS:
        if metric_col not in combined_df.columns:
            raise ValueError(f"Metric '{metric_col}' not found in CSV columns.")

        # Create simplified dataframe for plotting
        plot_df = combined_df[[
            "keyword", metric_col, "key_type", "language", "model", "template", "threshold"
        ]]

        # Plot per threshold (flat structure with threshold in filename)
        for threshold in sorted(plot_df["threshold"].unique()):
            df_threshold = plot_df[plot_df["threshold"] == threshold]

            # Save filtered plot_df
            output_csv_path = output_path / f"aggregated_plot_data_{metric_col.lower()}_thr_{threshold}.csv"
            df_threshold.to_csv(output_csv_path, index=False)
            print(f"📄 Saved aggregated data for {metric_col} threshold {threshold} to {output_csv_path}")

            # Generate plots with threshold embedded in filename
            plot_boxplot_distribution(
                df_threshold,
                metric_col,
                output_path / f"{metric_col.lower()}_boxplot_by_model_thr_{threshold}.pdf"
            )

            plot_template_effect(
                df_threshold,
                metric_col,
                output_path / f"{metric_col.lower()}_template_effect_thr_{threshold}.pdf"
            )

            plot_heatmap_summary(
                df_threshold,
                metric_col,
                output_path / f"{metric_col.lower()}_heatmap_model_language_thr_{threshold}.pdf"
            )

            plot_heatmap_summary_by_keytype_template(
                df=plot_df,
                metric=metric_col,
                output_file=output_path / f"{metric_col.lower()}_heatmap_full_thr_{threshold}.pdf"
            )

        # Generate threshold summary heatmap (one grid across all thresholds)
        plot_heatmap_by_threshold(
            plot_df,
            metric_col,
            output_path / f"{metric_col.lower()}_heatmap_model_language_by_threshold.pdf"
        )


if __name__ == "__main__":
    main()
