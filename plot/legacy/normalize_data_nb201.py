import os
import pickle
import glob
import numpy as np
import pandas as pd


def min_max_values(dataset, device_metric, filename):
    with open("benchmarks/nb201/benchmark_all_hw_metrics.pkl", "rb") as f:
        data = pickle.load(f)

    true_errors = []
    true_latencies = []
    archs_true = []

    for arch in data.keys():
        true_errors.append(100 - data[arch][dataset])
        true_latencies.append(data[arch][device_metric])
        archs_true.append(arch)

    max_lat, min_lat = max(true_latencies), min(true_latencies)
    max_err, min_err = max(true_errors), min(true_errors)

    print(max_lat, min_lat)
    print(max_err, min_err)

    df = pd.read_csv(filename)
    df["error"] = (df["error"] - min_err) / (max_err - min_err)
    df["latency"] = (df["latency"] - min_lat) / (max_lat - min_lat)
    # df = df[["latency", "error"]]
    print(filename)
    print(df.head())
    df.to_csv(filename, index=False)


base_directory = "ablations/baselines_poster_minimal/Qwen2.5-7B-Instruct-AWQ/latency"
file_names = glob.glob(f"{base_directory}/**/*.csv", recursive=True)
for file_path in file_names:
    directory, filename = os.path.split(
        file_path
    )  # Separate the directory and filename

    print(filename)

    dataset = "cifar10"

    if "titanx" in filename:
        device_metric = "titanx_256_latency"
    elif "1080" in filename:
        device_metric = "1080ti_32_latency"
    elif "fpga" in filename:
        device_metric = "fpga_latency"
    else:
        print("error")
        break
    print(device_metric)
    min_max_values(dataset, device_metric, file_path)
