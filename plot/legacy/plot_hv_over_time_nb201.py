import glob
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from HOLLM.utils.plot.plot_utils_nb201 import (
    get_result_file_names,
    extract_metadata_from_result_file,
    normalize_data,
    get_min_max_metrics_from_observed_data,
)

from HOLLM.utils.plot.plot_settings import get_visuals


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


def compute_hypervolume(df, reference_point):
    objectives = df.to_numpy()
    ind = HV(ref_point=reference_point)
    hypervolume = ind(objectives)
    return hypervolume


def convert_data_to_hv_over_time(fvals):
    hypervolume = []
    for step in range(1, len(fvals) + 1):
        # hv = compute_hypervolume(fvals.iloc[:step], [max_latency, max_perplexity])
        hv = compute_hypervolume(fvals.iloc[:step], [1.0, 1.0])
        hypervolume.append(hv)

    return hypervolume


def create_hv_over_time_plot(model_name, data, img_path):
    # plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(10, 7))

    for baseline_key, (hypervolume, hypervolume_std) in data.items():
        plot_mapping = get_visuals(baseline_key)
        print(f"plot_mapping {plot_mapping}")
        hypervolume = np.array(hypervolume)
        hypervolume_std = np.array(hypervolume_std)

        plt.plot(
            hypervolume,
            color=plot_mapping["color"],
            label=plot_mapping["label"],
            marker=plot_mapping["marker"],
            linestyle=plot_mapping["linestyle"],
            linewidth=2,
            markevery=5,
            markersize=8,
        )
        # plt.plot(len(hypervolume) - 1, hypervolume[-1], 'o')  # 'ro' means red circle marker
        plt.fill_between(
            range(len(hypervolume)),
            hypervolume - hypervolume_std,
            hypervolume + hypervolume_std,
            color=plot_mapping["color"],
            alpha=0.2,
        )
        # plt.text(len(hypervolume) - 1, hypervolume[-1], f"  ({len(hypervolume) - 1}, {hypervolume[-1]})", va='bottom', ha='right')

    ax.set_xlabel("Trials", fontsize=16)
    ax.set_ylabel("Hypervolume", fontsize=16)
    ax.set_title(
        "Hypervolume over Time for mooLLM on NB201 (titanx_256)",
        fontsize=18,
    )
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))
    ax.legend(handles, labels, frameon=True, edgecolor="black")
    # ax.set_ylim([0.85, 0.97])
    ax.set_ylim([0.65, 0.97])
    ax.set_xlim([0, 90])
    ax.grid(True, linestyle=":", linewidth=1)
    ax.set_xticks([5, 6, 7, 20, 40, 60, 80, 90])

    fig.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.savefig(f"{img_path}.pdf", dpi=300, bbox_inches="tight")


# Extract the data from the result files
# {"baseline_metric_device": dataframe containing the perplexity and hw_metric values for the evaluation}
def extract_data_from_experiment_files(filenames):
    data = {}
    for i, filepath in enumerate(filenames):
        baseline, metric, device, seed = extract_metadata_from_result_file(filepath)
        fvals = pd.read_csv(filepath)
        fvals = fvals[["error", "latency"]]

        key_name = f"{baseline}_{metric}_{device}"

        if key_name not in data.keys():
            data[key_name] = []
        data[key_name].append(fvals)

    return data


# Convert the results into the hypervolume over time
# {"baseline_metric_device": array containing the mean hypervolume per time step across the different seeds}
def convert_results_to_hypervolume_over_time(data, min_max_metrics_statistics):
    data_hypervolumes = {}
    for baseline_key, values in data.items():
        try:
            print(f"baseline key {baseline_key}")
            device = "_".join(baseline_key.split("_")[2:])
            metric = baseline_key.split("_")[1]
            metrics_key = "_".join(baseline_key.split("_")[1:])
            hypervolumes_over_time = []

            # min_max_metrics_surrogate = get_min_max_metrics_from_surrogate_data(hw_metric=metric, device=device)
            min_max_metrics = min_max_metrics_statistics[metrics_key]
            # print(f"min_max_metrics surrogate {min_max_metrics_surrogate}")
            print(f"min_max_metrics {min_max_metrics}")
            for fvals in values:
                # normalize_data(fvals, min_max_metrics)
                hv_over_time = convert_data_to_hv_over_time(fvals)
                hypervolumes_over_time.append(hv_over_time[:100])
            hypervolumes_over_time = np.array(hypervolumes_over_time)
            hypervolumes_over_time_mean = list(np.mean(hypervolumes_over_time, axis=0))
            hypervolumes_over_time_std = list(np.std(hypervolumes_over_time, axis=0))
            # add std for uncertainty bands

            data_hypervolumes[baseline_key] = (
                hypervolumes_over_time_mean,
                hypervolumes_over_time_std,
            )

        except Exception as e:
            print(e)
            print(f"device not found {device}")

    return data_hypervolumes


# create metric_pairs -> metric_device
def get_metrics_devices_pairs(data):
    devices = set()
    metrics = set()
    for baseline_key, _ in data.items():
        print(baseline_key)
        device = "_".join(baseline_key.split("_")[2:])
        print(f"device {device}")
        metric = baseline_key.split("_")[1]

        devices.add(device)
        metrics.add(metric)

    metrics_devices_pairs = []
    for m in metrics:
        for d in devices:
            metric_device_pair = f"{m}_{d}"
            metrics_devices_pairs.append(metric_device_pair)

    print(devices)
    print(metrics)
    print(metrics_devices_pairs)

    return metrics_devices_pairs


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
    args = parser.parse_args()

    base_directory = args.data_path
    save_directory = args.save_dir_name

    # base_directory = "./baselines/HWGPT/m/test_1"
    result_file_names_random = glob.glob(f"{base_directory}/**/*.csv", recursive=True)

    data = extract_data_from_experiment_files(result_file_names_random)
    metrics_devices_pairs = get_metrics_devices_pairs(data)

    data_1 = []
    for device_pair in metrics_devices_pairs:
        data_per_pair = {}
        keys = [k for k in data.keys() if device_pair in k]
        print(f"keys {keys}")

        for key in keys:
            data_per_pair[key] = data[key]

        data_1.append((device_pair, data_per_pair))

    min_max_values_metrics = get_min_max_metrics_from_observed_data(data_1)

    data_hypervolumes = convert_results_to_hypervolume_over_time(
        data, min_max_values_metrics
    )

    # Group data per experiment -> metric_device for all baselines
    # keys -> LSBO_metric_device_pair, RSBO_metric_device_pair, ...
    # data -> [(metric_device_pair, data_per_pair)]
    # data_per_pair -> {"RSBO_metric_device_pair": hypervolume_over_time, "LSBO_metric_device_pair": hypervolume_over_time, ...}
    data_hypervolumes_keys = list(data_hypervolumes.keys())

    data = []
    for device_pair in metrics_devices_pairs:
        data_per_pair = {}
        keys = [k for k in data_hypervolumes_keys if device_pair in k]
        print(f"keys {keys}")

        for key in keys:
            data_per_pair[key] = data_hypervolumes[key]

        data.append((device_pair, data_per_pair))

    # for each metric_pair create a plot
    # save it in a metric_pair named directory
    # experiment -> metric_device
    for experiment, data_per_metric_device_pair in data:
        directory_path = f"./ablation_plots/hypervolume_over_time/{save_directory}"
        experiment_name = f"{experiment}"
        save_path = f"{directory_path}/{experiment_name}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        create_hv_over_time_plot(
            experiment,
            data_per_metric_device_pair,
            save_path,
        )


if __name__ == "__main__":
    main()
