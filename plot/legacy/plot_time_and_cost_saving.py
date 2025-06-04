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
    "baselines/NB201/gpt-4o-mini-parallel-no-metrics/latency/cost_per_request/gpt-4o-mini-parallel-no-metrics_titanx-256_latency_42.csv"
)
sequential_df = pd.read_csv(
    "baselines/NB201/gpt-4o-mini-sequential-no-metrics/latency/cost_per_request/gpt-4o-mini-sequential-no-metrics_titanx-256_latency_42.csv"
)
parallel_surrogate_df = pd.read_csv(
    "baselines/NB201/gpt-4o-mini-parallel-surrogate-no-metrics/latency/cost_per_request/gpt-4o-mini-parallel-surrogate-no-metrics_titanx-256_latency_42.csv"
)

# Calculate costs
seq_prompt_cost = sequential_df["prompt_cost"].sum()
seq_completion_cost = sequential_df["completion_cost"].sum()
seq_total_cost = sequential_df["total_cost"].sum()

par_prompt_cost = parallel_df["prompt_cost"].sum()
par_completion_cost = parallel_df["completion_cost"].sum()
par_total_cost = parallel_df["total_cost"].sum()

par_surrogate_prompt_cost = parallel_surrogate_df["prompt_cost"].sum()
par_surrogate_completion_cost = parallel_surrogate_df["completion_cost"].sum()
par_surrogate_total_cost = parallel_surrogate_df["total_cost"].sum()

# Read time data
parallel_df_time = pd.read_csv(
    "baselines/NB201/gpt-4o-mini-parallel-no-metrics/latency/time_taken_per_trials/gpt-4o-mini-parallel-no-metrics_titanx-256_latency_42.csv"
)
sequential_df_time = pd.read_csv(
    "baselines/NB201/gpt-4o-mini-sequential-no-metrics/latency/time_taken_per_trials/gpt-4o-mini-sequential-no-metrics_titanx-256_latency_42.csv"
)
parallel_surrogate_df_time = pd.read_csv(
    "baselines/NB201/gpt-4o-mini-parallel-surrogate-no-metrics/latency/time_taken_per_trials/gpt-4o-mini-parallel-surrogate-no-metrics_titanx-256_latency_42.csv"
)

# Aggregate time
seq_candidate_sampler = sequential_df_time["candidate_sampler"].sum() / 60
seq_surrogate_model = sequential_df_time["surrogate_model"].sum() / 60

par_candidate_sampler = parallel_df_time["candidate_sampler"].sum() / 60
par_surrogate_model = parallel_df_time["surrogate_model"].sum() / 60

par_candidate_sampler_surrogate = (
    parallel_surrogate_df_time["candidate_sampler"].sum() / 60
)
par_surrogate_model_surrogate = parallel_surrogate_df_time["surrogate_model"].sum() / 60

# Data for plotting
methods = ["Sequential", "Parallel (CS)", "Parallel (CS + SM)"]
x = np.arange(len(methods)) * 1.7  # Add spacing between cost/time pairs
offset = 0.4
bar_width = 0.7

cost_colors = ["#4e79a7", "#e15759"]  # Cost colors
time_colors = ["#298c8c", "#f2c45f"]  # Time colors

fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

# Cost bars (left axis)
ax1.bar(
    x - offset,
    [seq_completion_cost, par_completion_cost, par_surrogate_completion_cost],
    width=bar_width,
    color=cost_colors[0],
    label="Completion Cost",
    edgecolor="black",
)
ax1.bar(
    x - offset,
    [seq_prompt_cost, par_prompt_cost, par_surrogate_prompt_cost],
    width=bar_width,
    color=cost_colors[1],
    label="Prompt Cost",
    edgecolor="black",
    bottom=[seq_completion_cost, par_completion_cost, par_surrogate_completion_cost],
)

# Time bars (right axis)
ax2 = ax1.twinx()
ax2.bar(
    x + offset,
    [seq_surrogate_model, par_surrogate_model, par_surrogate_model_surrogate],
    width=bar_width,
    color=time_colors[0],
    label="Surrogate Model Time",
    edgecolor="black",
)
ax2.bar(
    x + offset,
    [seq_candidate_sampler, par_candidate_sampler, par_candidate_sampler_surrogate],
    width=bar_width,
    color=time_colors[1],
    label="Candidate Sampler Time",
    edgecolor="black",
    bottom=[seq_surrogate_model, par_surrogate_model, par_surrogate_model_surrogate],
)

# Axis formatting
ax1.set_ylabel("Cost (USD)", fontweight="bold")
ax2.set_ylabel("Time (minutes)", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(methods, ha="center")
ax1.spines[["top", "right"]].set_visible(False)
ax2.spines[["top", "left"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.3)


# Adding text labels
def add_labels(ax, bars, values, offset=0.2, currency=False):
    for bar, value in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2
        label = f"${value:.2f}" if currency else f"{value:.1f}m"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )


# Apply text labels
for bars, values, is_cost in [
    (
        ax1.patches[:3],
        [seq_completion_cost, par_completion_cost, par_surrogate_completion_cost],
        True,
    ),
    (
        ax1.patches[3:],
        [seq_prompt_cost, par_prompt_cost, par_surrogate_prompt_cost],
        True,
    ),
    (
        ax2.patches[:3],
        [seq_surrogate_model, par_surrogate_model, par_surrogate_model_surrogate],
        False,
    ),
    (
        ax2.patches[3:],
        [seq_candidate_sampler, par_candidate_sampler, par_candidate_sampler_surrogate],
        False,
    ),
]:
    add_labels(ax1 if is_cost else ax2, bars, values, currency=is_cost)

# Total labels above bars
for i, (total_cost, total_time) in enumerate(
    zip(
        [seq_total_cost, par_total_cost, par_surrogate_total_cost],
        [
            seq_candidate_sampler + seq_surrogate_model,
            par_candidate_sampler + par_surrogate_model,
            par_candidate_sampler_surrogate + par_surrogate_model_surrogate,
        ],
    )
):
    ax1.text(
        x[i] - offset,
        total_cost + 0.1,
        f"${total_cost:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none"),
    )
    ax2.text(
        x[i] + offset,
        total_time + 1,
        f"{total_time:.1f}m",
        ha="center",
        va="bottom",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none"),
    )

# Legend
ax1.legend(loc="upper right", frameon=True)
# Create a combined legend for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    frameon=True,
)

plt.tight_layout()

# Save plot
output_dir = "plots/combined"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(
    f"{output_dir}/combined_cost_time_comparison.png", dpi=300, bbox_inches="tight"
)
plt.savefig(
    f"{output_dir}/combined_cost_time_comparison.pdf", dpi=300, bbox_inches="tight"
)
plt.close()
