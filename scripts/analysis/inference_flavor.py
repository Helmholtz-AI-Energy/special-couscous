import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from specialcouscous.utils.slurm import time_to_seconds

pd.set_option("display.max_rows", None)

# Get the root directory where results are stored from command line.
root_dir = sys.argv[1]
data_set = root_dir.split(os.sep)[-1]
# Dictionary to store the values grouped by (dataset, number_of_tasks, dataseed)
results = defaultdict(list)

# The specific column you want to average in the CSV files
target_columns = ["time_sec_training", "time_sec_all-gathering_model", "time_sec_test"]
# Walk through the directory structure to find CSV files
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".csv"):
            # Extract relevant information from the directory structure.
            parts = dirpath.split(os.sep)
            number_of_tasks = int(parts[-2].split("_")[1])
            data_seed = int(parts[-1].split("_")[1])  # Extract data seed from path.
            model_seed = int(parts[-1].split("_")[2])  # Extract model seed from path.
            print(
                f"{data_set}: {number_of_tasks} tasks, data seed {data_seed}, model seed {model_seed}"
            )
            # Read the CSV file into a pandas dataframe.
            file_path = os.path.join(dirpath, filename)
            df = pd.read_csv(file_path)

            # Extract the value from the target column and store it
            # Parallel runs:
            for column in target_columns:
                if column in df.columns:
                    time_in_sec = df.loc[df["comm_rank"] == "global", column].values[0]
                    print(f"{column}: {time_in_sec} s")
                else:
                    time_in_sec = np.nan
                results[(data_set, number_of_tasks, data_seed, model_seed)].append(
                    time_in_sec
                )
        if filename.endswith(".out"):
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
            results[(data_set, number_of_tasks, data_seed, model_seed)].append(
                wall_clock_time
            )

print(results)
# # Save the results to a pandas dataframe.
# results_df = pd.DataFrame(
#     [(k[0], k[1], k[2], k[3], v[0]) for k, v in results.items()],
#     columns=[
#         "Dataset",
#         "Number of tasks",
#         "Data seed",
#         "Model seed",
#         "Time to train",
#         "Time to all-gather shared global model",
#         "Time to test"
#     ],
# )
# results_df = results_df.sort_values(by=["Number of tasks", "Data seed", "Model seed"])
#
# avg_data_seeds = (
#     results_df.groupby(["Number of tasks", "Data seed"])
#     .agg({"Global test accuracy": "mean"})
#     .reset_index()
# )
# print(avg_data_seeds)
# avg_n_tasks = (
#     results_df.groupby(["Number of tasks"])
#     .agg({"Global test accuracy": "mean"})
#     .reset_index()
# )
# print(avg_n_tasks)
#
# plt.figure(figsize=(10, 6))
# plt.grid(True)
#
# # Plot individual test accuracy vs number of tasks as small dots
# plt.scatter(
#     [str(n_tasks) for n_tasks in results_df["Number of tasks"]],
#     results_df["Global test accuracy"] * 100,
#     label="Individual test accuracies",
#     marker=".",
#     c=results_df["Data seed"],
#     cmap="winter",
#     zorder=10,
#     alpha=0.5,
# )
#
# # Plot average test accuracy over model seeds for each data seed vs. number of tasks as larger points
# plt.scatter(
#     [str(n_tasks) for n_tasks in avg_data_seeds["Number of tasks"]],
#     avg_data_seeds["Global test accuracy"] * 100,
#     label="Average over model seeds for each data seed",
#     s=200,
#     marker="_",
#     c=avg_data_seeds["Data seed"],
#     cmap="winter",
#     alpha=0.5,
# )
#
# # Plot overall average test accuracy vs number of tasks as stars
# plt.scatter(
#     [str(n_tasks) for n_tasks in avg_n_tasks["Number of tasks"]],
#     avg_n_tasks["Global test accuracy"] * 100,
#     label="Average over all seeds for each number of tasks",
#     s=200,
#     marker="*",
#     facecolor="none",
#     edgecolor="firebrick",
#     linewidths=1.3,
#     # color="firebrick",
#     zorder=20,
# )
#
# # Add labels and legend
# plt.xlabel("Number of tasks", fontweight="bold")
# plt.ylabel("Test accuracy / %", fontweight="bold")
# data_set = data_set.replace("_", "")
# plt.title(
#     f"Robust seeds {data_set}: Global test accuracy vs. number of tasks",
#     fontweight="bold",
# )
# plt.legend(loc="lower left", fontsize="small")
# # plt.legend(loc='upper left', bbox_to_anchor=(0.5, 0.82), fontsize="small")
# plt.savefig(pathlib.Path(root_dir) / f"{data_set}_acc_drop.png")
#
# # Show the plot
# plt.show()
