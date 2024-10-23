import os
import pathlib
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from specialcouscous.utils.slurm import time_to_seconds

pd.set_option("display.max_rows", None)

# Get the root directory where results are stored from command line.
root_dir = sys.argv[1]
data_set = root_dir.split(os.sep)[-1]
flavor = root_dir.split(os.sep)[-2].replace("_", " ")
# Dictionary to store the values grouped by (dataset, number_of_tasks, dataseed)
results = defaultdict(list)

# The specific column you want to average in the CSV files
# target_columns = ["accuracy_global_test", "accuracy_test"]
target_columns = ["accuracy_global_test", "accuracy_local_test", "accuracy_test"]

# Walk through the directory structure to find CSV files
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".csv"):
            # Extract relevant information from the directory structure
            parts = dirpath.split(os.sep)
            number_of_tasks = int(parts[-2].split("_")[1])
            model_seed = int(parts[-1].split("_")[2])  # Extract model seed from path.
            print(f"{data_set}: {number_of_tasks} tasks, model seed {model_seed}")
            # Read the CSV file into a pandas dataframe.
            file_path = os.path.join(dirpath, filename)
            print(file_path)
            df = pd.read_csv(file_path)

            # Extract the value from the target column and store it
            # Parallel runs:
            if "accuracy_global_test" in df.columns:
                global_test_accuracy = df.loc[
                    df["comm_rank"] == "global", "accuracy_global_test"
                ].values[0]

            if "accuracy_test" in df.columns:
                global_test_accuracy = df["accuracy_test"].values[0]

            print(f"Global test accuracy: {global_test_accuracy}")
            results[(data_set, number_of_tasks, model_seed)].append(
                global_test_accuracy
            )

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".out"):
            parts = dirpath.split(os.sep)
            number_of_tasks = int(parts[-2].split("_")[1])
            model_seed = int(parts[-1].split("_")[2])  # Extract model seed from path.
            pattern_wall_clock_time = (
                r"Job Wall-clock time: (\d+:\d+:\d+|\d+-\d+:\d+:\d+)"
            )
            with open(
                os.path.join(dirpath, filename), "r"
            ) as file:  # Load input text from the file.
                input_text = file.read()
            # Extract wall-clock time.
            wall_clock_time = time_to_seconds(
                re.search(pattern_wall_clock_time, input_text).group(1)  # type:ignore
            )
            print(f"Wall-clock time: {wall_clock_time} s")
            results[(data_set, number_of_tasks, model_seed)].append(wall_clock_time)

# Save the results to a pandas dataframe.
results_df = pd.DataFrame(
    [(k[0], k[1], k[2], v[0], v[1]) for k, v in results.items()],
    columns=[
        "Dataset",
        "Number of nodes",
        "Model seed",
        "Global test accuracy",
        "Wall-clock time",
    ],
)
results_df = results_df.sort_values(by=["Number of nodes", "Model seed"])

# For each parallelization level, get average of test accuracy over model seeds.
avg_acc_n_tasks = (
    results_df.groupby(["Number of nodes"])
    .agg({"Global test accuracy": "mean"})
    .reset_index()
)
# For each parallelization level, get average of wall-clock time over model seeds.
avg_time_n_tasks = (
    results_df.groupby(["Number of nodes"])
    .agg({"Wall-clock time": "mean"})
    .reset_index()
)
print(avg_time_n_tasks)
print(avg_time_n_tasks["Wall-clock time"][avg_time_n_tasks["Number of nodes"] == 1])

# Create the figure and the first axis for test accuracy
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
# Set title
data_set = data_set.replace("_", "")
plt.suptitle(
    f"Weak scaling {data_set} {flavor}",
    fontweight="bold",
)
# Plot individual test accuracy vs number of tasks (left y-axis)
ax1.scatter(
    [str(n_tasks) for n_tasks in results_df["Number of nodes"]],
    results_df["Global test accuracy"] * 100,
    label="Individual test accuracies",
    marker=".",
    color="k",
    zorder=10,
    alpha=0.5,
)

# Plot overall average test accuracy vs number of tasks as stars (left y-axis)
ax1.scatter(
    [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
    avg_acc_n_tasks["Global test accuracy"] * 100,
    label="Average over all model seeds",
    s=80,
    marker="X",
    facecolor="none",
    edgecolor="k",
    linewidths=1.3,
    zorder=20,
)

# Customize the left y-axis (test accuracy)
ax1.set_ylabel("Test accuracy / %", fontweight="bold")
ax1.grid(True)
ax1.legend(loc="lower right", fontsize="small")

# Plot wall-clock time vs number of tasks (right y-axis)
ax2.scatter(
    [str(n_tasks) for n_tasks in results_df["Number of nodes"]],
    results_df["Wall-clock time"] / 60,  # Assuming 'Wall-clock time' column exists
    label="Individual wall-clock times",
    marker=".",
    color="k",
    zorder=5,
    alpha=0.5,
)

ax2.scatter(
    [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
    avg_time_n_tasks["Wall-clock time"]
    / 60,  # Assuming 'Wall-clock time' column exists
    label="Average over all model seeds",
    s=80,
    marker="X",
    facecolor="none",
    edgecolor="k",
    linewidths=1.3,
    zorder=20,
)
ax2.set_ylabel("Wall-clock time / min", fontweight="bold")
ax2.legend(loc="upper left", fontsize="small")
ax2.grid(True)

print(
    avg_time_n_tasks["Wall-clock time"][avg_time_n_tasks["Number of nodes"] == 1].values
    / avg_time_n_tasks["Wall-clock time"]
)
# Plot overall average wall-clock time (right y-axis)
ax3.scatter(
    [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
    avg_time_n_tasks["Wall-clock time"][avg_time_n_tasks["Number of nodes"] == 1].values
    / avg_time_n_tasks["Wall-clock time"],  # Assuming 'Wall-clock time' column exists
    label="Efficiency",
    s=80,
    marker="X",
    facecolor="none",
    edgecolor="k",
    linewidths=1.3,
    zorder=15,
)

# Customize the right y-axis (wall-clock time)
ax3.set_ylabel("Efficiency", fontweight="bold")
ax3.set_xlabel("Number of nodes", fontweight="bold")
ax3.grid(True)

# Ensure the layout is tight so labels don't overlap
plt.tight_layout()

# Save the figure
flavor = flavor.replace(" ", "_")
plt.savefig(pathlib.Path(root_dir) / f"{data_set}_{flavor}_weak_scaling.png")

# Show the plot
plt.show()
