import os
import glob
import argparse

import pandas as pd
from paretoset import paretoset
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from mooLLM.utils.plot.extract_statistics_from_surrogate import (
    get_min_max_metrics_from_surrogate_data,
)
from mooLLM.utils.plot.plot_utils import extract_metadata_from_result_file

from mooLLM.utils.plot.plot_settings import get_visuals
import matplotlib.colors as mcolors
import numpy as np

plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titlesize": 14,
    }
)


def max_min_norm(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def plot_pareto_set(observed_fvals, img_path, title, dataset, device_metric):
    fig, ax = plt.subplots(figsize=(10, 7))

    i = 0
    for filename, fvals in observed_fvals.items():
        plot_mapping = get_visuals(filename)
        print(f"plot_mapping {plot_mapping}")
        i += 1
        fvals = fvals[["perplexity", "hw_metric"]]

        mask = paretoset(fvals, sense=["min", "min"])

        pareto_df = fvals[mask]
        non_pareto_df = fvals[~mask]

        # optimal_pareto_df = get_optimal_pareto_set(dataset, device_metric)

        dot_size = 200
        # Plot the data
        # plot non_pareto_df
        is_priority = any(x in filename.lower() for x in ["gpt", "rs", "nsga2", "rsbo"])
        zorder_value = 5 if is_priority else 1

        # plt.scatter(
        #     non_pareto_df["hw_metric"],
        #     non_pareto_df["perplexity"],
        #     alpha=0.2,
        #     s=dot_size,
        #     color=plot_mapping["color"],
        #     marker=plot_mapping["marker"],
        #     zorder=zorder_value,
        # )

        # -------------------------------
        # Plot non-Pareto points with increasing alpha
        # -------------------------------
        if len(non_pareto_df) > 0:
            # Sort non-Pareto points by their original order (assumed chronological)
            non_pareto_df = non_pareto_df.sort_index().reset_index(drop=True)
            n_points = len(non_pareto_df)

            # Define the minimum and maximum alpha values.
            min_alpha = 0.1  # older points (less opaque)
            max_alpha = 0.4  # newer points (more opaque)

            # Create a linearly spaced array of alpha values.
            alphas = np.linspace(min_alpha, max_alpha, n_points)

            # Convert the base color (from the mapping) to an RGBA tuple.
            base_color = mcolors.to_rgba(plot_mapping["color"])

            # Generate a list of colors with varying alpha values.
            colors = [
                (base_color[0], base_color[1], base_color[2], alpha) for alpha in alphas
            ]

            # Plot each non-Pareto point with its corresponding color.
            plt.scatter(
                non_pareto_df["hw_metric"],
                non_pareto_df["perplexity"],
                s=dot_size,
                color=colors,
                marker=plot_mapping["marker"],
            )

        # plot pareto_df
        plt.scatter(
            pareto_df["hw_metric"],
            pareto_df["perplexity"],
            alpha=0.9,
            s=dot_size,
            label=plot_mapping["label"],
            color=plot_mapping["color"],
            marker=plot_mapping["marker"],
            zorder=zorder_value,
        )
        if len(pareto_df) > 1:
            pareto_df_sorted = pareto_df.sort_values("hw_metric")
            plt.plot(
                pareto_df_sorted["hw_metric"],
                pareto_df_sorted["perplexity"],
                color=plot_mapping["color"],
                alpha=0.9,
                linewidth=2,
                linestyle="-",
            )
        # plot optimal pareto
        # plt.scatter(optimal_pareto_df['latency'], optimal_pareto_df['score'], alpha=0.3, s=dot_size, label='Optimal Pareto Set', color='green')

    ax.set_xlabel("Latency", fontsize=16)
    ax.set_ylabel("Perplexity", fontsize=16)
    ax.set_title("Sampling behavior of mooLLM on HWGPT vs baselines", fontsize=18)
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))
    ax.legend(handles, labels, frameon=True, edgecolor="black")

    # ax.set_xlim([-0.1, 1.1])
    # ax.set_ylim([-0.1, 1.1])
    ax.grid(True, linestyle=":", linewidth=1)
    fig.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.savefig(f"{img_path}.pdf", dpi=300, bbox_inches="tight")


def get_result_file_names(folder_path, search_pattern):
    # Construct the search pattern
    search_pattern = os.path.join(folder_path, search_pattern)

    # Use glob to find all files that match the pattern
    files = glob.glob(search_pattern)

    # Extract the file names from the full paths
    result_file_names = [
        os.path.join(folder_path, os.path.basename(file)) for file in files
    ]

    return result_file_names


def main():
    parser = argparse.ArgumentParser(description="Run with a specific model.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./baselines/HWGPT/m/test",
        help="The directory where the experiment data is saved",
    )
    parser.add_argument(
        "--save_dir_name",
        type=str,
        default="test",
        help="The name of directory where the experiment data is saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for which the plots should be created (default: 42)",
    )
    args = parser.parse_args()

    # base_directory = "./baselines/HWGPT/m/test"
    base_directory = args.data_path
    save_directory = args.save_dir_name
    seed = args.seed
    result_file_names_random = glob.glob(f"{base_directory}/**/*.csv", recursive=True)

    result_file_names_random = [
        file
        for file in result_file_names_random
        if os.path.basename(file).split("_")[-1].split(".")[0] == str(seed)
    ]

    # take the data per seed

    data = {}
    for filepath in result_file_names_random:
        baseline, metric, device, seed = extract_metadata_from_result_file(filepath)
        key_name = f"{baseline}_{metric}_{device}"
        fvals = pd.read_csv(filepath)
        fvals = fvals[["perplexity", "hw_metric"]][:100]

        key_name = f"{baseline}_{metric}_{device}"

        if key_name not in data.keys():
            data[key_name] = []
        data[key_name] = fvals

    # separate the data for each experiment

    # create metric_pairs -> metric_device
    def get_metrics_devices_pairs(data):
        devices = set()
        metrics = set()
        for baseline_key, _ in data.items():
            device = "_".join(baseline_key.split("_")[2:])
            metric = baseline_key.split("_")[1]

            devices.add(device)
            metrics.add(metric)

        metrics_devices_pairs = []
        for m in metrics:
            for d in devices:
                metric_device_pair = f"{m}_{d}"
                metrics_devices_pairs.append(metric_device_pair)

        return metrics_devices_pairs

    metrics_devices = get_metrics_devices_pairs(data)
    # fvals_total
    # {"model_name" -> pd dataframe fvals}

    metrics_devices_pairs = get_metrics_devices_pairs(data)

    data_keys = list(data.keys())
    print(data_keys)

    final_data = []
    for device_pair in metrics_devices_pairs:
        data_per_pair = {}
        keys = [k for k in data_keys if device_pair in k]
        print(f"keys {keys}")

        for key in keys:
            data_per_pair[key] = data[key]

        final_data.append((device_pair, data_per_pair))

    # for each metric_pair create a plot
    # save it in a metric_pair named directory

    # experiment -> metric_device
    for experiment, data_per_metric_device_pair in final_data:
        directory_path = f"./ablation_plots_hwgpt/pareto_set/{save_directory}"
        experiment_name = f"{experiment}_seed_{seed}"
        save_path = f"{directory_path}/{experiment_name}"

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        plot_pareto_set(
            data_per_metric_device_pair,
            save_path,
            experiment,
            "",
            "",
        )


if __name__ == "__main__":
    main()
