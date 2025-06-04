import os
import glob
import argparse
import numpy as np
import pickle
import pandas as pd
from paretoset import paretoset
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from mooLLM.utils.plot.plot_utils_nb201 import extract_metadata_from_result_file
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


def get_optimal_pareto_set(dataset, device_metric):
    with open("benchmarks/nb201/benchmark_all_hw_metrics.pkl", "rb") as f:
        data = pickle.load(f)

    true_errors = []
    true_latencies = []
    archs_true = []

    for arch in data.keys():
        true_errors.append(100 - data[arch][dataset])
        true_latencies.append(data[arch][device_metric])
        archs_true.append(arch)

    true_errors_filtered = true_errors
    true_latencies_filtered = true_latencies

    true_errors_filtered = np.array(true_errors_filtered)
    true_latencies_filtered = np.array(true_latencies_filtered)

    observed_fvals = pd.DataFrame(
        {"error": true_errors_filtered, "latency": true_latencies_filtered}
    )

    max_lat, min_lat = max(true_latencies), min(true_latencies)
    max_err, min_err = max(true_errors), min(true_errors)
    observed_fvals["error"] = (observed_fvals["error"] - min_err) / (max_err - min_err)
    observed_fvals["latency"] = (observed_fvals["latency"] - min_lat) / (
        max_lat - min_lat
    )

    # observed_fvals = normalize_data(observed_fvals, dataset, device_metric)
    mask = paretoset(observed_fvals, sense=["min", "min"])
    pareto_df = observed_fvals[mask]

    return pareto_df


def max_min_norm(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def plot_pareto_set(observed_fvals, img_path, title, dataset, device_metric):
    fig, ax = plt.subplots(figsize=(10, 7))

    dot_size = 200

    # plot optimal pareto
    optimal_pareto_df = get_optimal_pareto_set(dataset, device_metric)
    plt.scatter(
        optimal_pareto_df["latency"],
        optimal_pareto_df["error"],
        s=dot_size,
        label="Optimal Pareto Set",
        color="black",
        zorder=-1,
    )

    i = 0
    for filename, fvals in observed_fvals.items():
        plot_mapping = get_visuals(filename)

        i += 1
        fvals = fvals[["error", "latency"]]

        mask = paretoset(fvals, sense=["min", "min"])

        pareto_df = fvals[mask]
        non_pareto_df = fvals[~mask]

        # Plot the data
        # plot non_pareto_df
        is_priority = any(x in filename.lower() for x in ["gpt"])
        zorder_value = 5 if is_priority else 1
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
                non_pareto_df["latency"],
                non_pareto_df["error"],
                s=dot_size,
                color=colors,
                marker=plot_mapping["marker"],
            )
        # plot pareto_df
        plt.scatter(
            pareto_df["latency"],
            pareto_df["error"],
            s=dot_size,
            alpha=0.9,
            label=plot_mapping["label"],
            color=plot_mapping["color"],
            marker=plot_mapping["marker"],
            zorder=zorder_value,
        )
        if len(pareto_df) > 1:
            pareto_df_sorted = pareto_df.sort_values("latency")
            plt.plot(
                pareto_df_sorted["latency"],
                pareto_df_sorted["error"],
                color=plot_mapping["color"],
                alpha=0.9,
                linewidth=2,
                linestyle="-",
            )

    ax.set_xlabel("Latency", fontsize=16)
    ax.set_ylabel("Error", fontsize=16)
    ax.set_title(
        "Sampling behavior of mooLLM on NB201 vs baselines (DeepSeek R1)", fontsize=18
    )
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))
    ax.legend(handles, labels, frameon=True, edgecolor="black")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 0.5])
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
        fvals = fvals[["error", "latency"]][:100]

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
        directory_path = f"./ablation_plots/pareto_set/{save_directory}"
        experiment_name = f"{experiment}_seed_{seed}"
        save_path = f"{directory_path}/{experiment_name}"

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        print(experiment)
        device_metric = None
        if "titanx" in experiment:
            device_metric = "titanx_256_latency"
        elif "1080" in experiment:
            device_metric = "1080ti_32_latency"
        elif "fpga" in experiment:
            device_metric = "fpga_latency"
        else:
            print("error")
            break

        plot_pareto_set(
            data_per_metric_device_pair,
            save_path,
            experiment,
            "cifar10",
            device_metric,
        )


if __name__ == "__main__":
    main()
