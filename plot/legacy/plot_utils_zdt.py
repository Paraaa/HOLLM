import os
import glob
import pandas as pd


def extract_metadata_from_result_file(filepath):
    filename = os.path.basename(filepath)
    path_parts = filepath.split("/")

    # Extract baseline and metric
    # baseline = path_parts[-3]  # Third last part of the path
    metric = path_parts[-2]  # Second last part of the path

    # Use regex to extract device and seed from the filename
    # LSBO_energies_cpu_amd_7452_31415927.csv
    parts = filename.split("_")

    baseline = parts[0]  # LSBO - baseline
    # device = parts[1]  # energies - metric
    seed = parts[-1].split(".")[0]  # 31415927 - seed
    device = "_".join(parts[1:-2])  # cpu_amd_7452 - device
    return baseline, metric, device, seed


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


def get_min_max_metrics_from_observed_data(data_1):
    min_max_values = {}
    for experiment, data_per_metric_device_pair in data_1:
        min_perplexity = 100000
        max_perplexity = -100
        min_hw_metric = 100000
        max_hw_metric = -100

        for baseline_key, dataframe_per_baseline in data_per_metric_device_pair.items():
            trimmed_dfs = [df.iloc[:100] for df in dataframe_per_baseline]
            all_dfs = pd.concat(trimmed_dfs)
            all_dfs_grouped = all_dfs.groupby(level=0)
            mean_df = all_dfs_grouped.mean()
            data_per_metric_device_pair[baseline_key] = mean_df

            current_max_perplexity = all_dfs["f1"].max()
            current_min_perplexity = all_dfs["f1"].min()

            current_max_hw_metric = all_dfs["f2"].max()
            current_min_hw_metric = all_dfs["f2"].min()

            min_perplexity = min(min_perplexity, current_min_perplexity)
            max_perplexity = max(max_perplexity, current_max_perplexity)
            min_hw_metric = min(min_hw_metric, current_min_hw_metric)
            max_hw_metric = max(max_hw_metric, current_max_hw_metric)

            if experiment not in min_max_values:
                min_max_values[experiment] = {}

            min_max_values[experiment] = {
                "min_perplexity": min_perplexity,
                "max_perplexity": max_perplexity,
                "min_hw_metric": min_hw_metric,
                "max_hw_metric": max_hw_metric,
            }
        print(f"max per experiment {max_perplexity}")
    print(min_max_values)
    return min_max_values


def normalize_data(fvals, min_max_metrics):
    min_hw_metric = min_max_metrics["min_hw_metric"]
    max_hw_metric = min_max_metrics["max_hw_metric"]
    min_perplexity = min_max_metrics["min_perplexity"]
    max_perplexity = min_max_metrics["max_perplexity"]
    # fvals columns -> Index(['Unnamed: 0', 'configs', 'runtime_traj', 'perplexity', 'hw_metric'], dtype='object')
    fvals["f2"] = (fvals["f2"] - min_hw_metric) / (max_hw_metric - min_hw_metric)
    fvals["f1"] = (fvals["f1"] - min_perplexity) / (max_perplexity - min_perplexity)
