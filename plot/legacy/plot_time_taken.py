import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure scientific plotting style
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


# Read the CSV files
parallel_df = pd.read_csv(
    "baselines/NB201/gpt-4o-mini-parallel-no-metrics/latency/time_taken_per_trials/gpt-4o-mini-parallel-no-metrics_titanx-256_latency_42.csv"
)

parallel_surrogate_df = pd.read_csv(
    "baselines/NB201/gpt-4o-mini-parallel-surrogate-no-metrics/latency/time_taken_per_trials/gpt-4o-mini-parallel-surrogate-no-metrics_titanx-256_latency_42.csv"
)

sequential_df = pd.read_csv(
    "baselines/NB201/gpt-4o-mini-sequential-no-metrics/latency/time_taken_per_trials/gpt-4o-mini-sequential-no-metrics_titanx-256_latency_42.csv"
)


# Aggregate total time taken for each part in each DataFrame
seq_candidate_sampler = sequential_df["candidate_sampler"].sum() / 60
seq_surrogate_model = sequential_df["surrogate_model"].sum() / 60

par_candidate_sampler = parallel_df["candidate_sampler"].sum() / 60
par_surrogate_model = parallel_df["surrogate_model"].sum() / 60

par_candidate_sampler_surrogate = parallel_surrogate_df["candidate_sampler"].sum() / 60
par_surrogate_model_surrogate = parallel_surrogate_df["surrogate_model"].sum() / 60

# Data preparation
methods = [
    "Sequential",
    "Parallel (CS)",
    "Parallel (CS + SM)",
]
candidate_sampler_times = [
    seq_candidate_sampler,
    par_candidate_sampler,
    par_candidate_sampler_surrogate,
]
surrogate_model_times = [
    seq_surrogate_model,
    par_surrogate_model,
    par_surrogate_model_surrogate,
]

total_times = [sum(x) for x in zip(candidate_sampler_times, surrogate_model_times)]

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

colors = ["#4e79a7", "#e15759"]  # Colorblind-friendly

# Plot bars with edge formatting
p1 = ax.bar(
    methods,
    surrogate_model_times,
    label="Surrogate Model",
    color=colors[0],
    edgecolor="black",
    linewidth=0.5,
)

p2 = ax.bar(
    methods,
    candidate_sampler_times,
    bottom=surrogate_model_times,
    label="Candidate Sampler",
    color=colors[1],
    edgecolor="black",
    linewidth=0.5,
)

# Formatting improvements
ax.set_ylabel("Time (minutes)", fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(
    frameon=True,
    framealpha=0.9,
    loc="upper right",
    ncol=1,
)

plt.xticks(rotation=45, ha="right")


# Improved text labels
def add_labels(bars1, bars2, times1, times2):
    for bar1, bar2, time1, time2 in zip(bars1, bars2, times1, times2):
        ax.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height() / 2,
            f"{time1:.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

        ax.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar1.get_height() + bar2.get_height() / 2,
            f"{time2:.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )


add_labels(p1, p2, surrogate_model_times, candidate_sampler_times)

# Total labels with background
for i, total in enumerate(total_times):
    ax.text(
        i,
        total + 1,
        f"Total: {total:.1f}m",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"),
    )

plt.tight_layout()

# Save in multiple formats
dir = "plots/time_taken_comparison"
os.makedirs(dir, exist_ok=True)
plt.savefig(f"{dir}/parallel_vs_sequential_test.png", dpi=300, bbox_inches="tight")
plt.close()
