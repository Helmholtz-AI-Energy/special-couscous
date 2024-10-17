import os
import pathlib
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_rows", None)

# Get the root directory where results are stored from command line.
root_dir = sys.argv[1]
data_set = root_dir.split(os.sep)[-1]
# Dictionary to store the values grouped by (dataset, number_of_tasks, dataseed)
results = defaultdict(list)

# The specific column you want to average in the CSV files
target_columns = ["accuracy_global_test", "accuracy_test"]

# Walk through the directory structure to find CSV files
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".csv"):
            # Extract relevant information from the directory structure
            print(dirpath)
            parts = dirpath.split(os.sep)
            number_of_tasks = int(parts[-2].split("_")[1])
            data_seed = int(parts[-1].split("_")[1])  # Extract data seed from path.
            model_seed = int(parts[-1].split("_")[2])  # Extract model seed from path.
            print(
                f"{data_set}: {number_of_tasks} tasks, data seed {data_seed}, model seed {model_seed}"
            )
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
                print(f"Global test accuracy: {global_test_accuracy}")
                results[(data_set, number_of_tasks, data_seed, model_seed)].append(
                    global_test_accuracy
                )
            if "accuracy_test" in df.columns:
                global_test_accuracy = df["accuracy_test"].values[0]
                print(f"Global test accuracy: {global_test_accuracy}")
                results[(data_set, number_of_tasks, data_seed, model_seed)].append(
                    global_test_accuracy
                )

print(results)

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".out"):
            parts = dirpath.split(os.sep)
            number_of_tasks = int(parts[-2].split("_")[1])
            data_seed = int(parts[-1].split("_")[1])  # Extract data seed from path.
            model_seed = int(parts[-1].split("_")[2])  # Extract model seed from path.
            pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
            with open(
                os.path.join(dirpath, filename), "r"
            ) as file:  # Load input text from the file.
                input_text = file.read()
                print(dirpath)
            energy_match = re.search(pattern_energy, input_text)
            energy_consumed = float(energy_match.group(0))  # type:ignore
            print(f"Energy Consumed: {energy_consumed:.2f} Watthours")
            results[(data_set, number_of_tasks, data_seed, model_seed)].append(
                energy_consumed
            )

print(results)

# Save the results to a pandas dataframe.
results_df = pd.DataFrame(
    [(k[0], k[1], k[2], k[3], v[0], v[1]) for k, v in results.items()],
    columns=[
        "Dataset",
        "Number of tasks",
        "Data seed",
        "Model seed",
        "Global test accuracy",
        "Energy consumed",
    ],
)
results_df = results_df.sort_values(by=["Number of tasks", "Data seed", "Model seed"])

overall_energy = results_df["Energy consumed"].sum()

avg_data_seeds = (
    results_df.groupby(["Number of tasks", "Data seed"])
    .agg({"Global test accuracy": "mean"})
    .reset_index()
)
print(avg_data_seeds)
avg_n_tasks = (
    results_df.groupby(["Number of tasks"])
    .agg({"Global test accuracy": "mean"})
    .reset_index()
)
print(avg_n_tasks)

f, ax = plt.subplots(figsize=(10, 6))
plt.grid(True)

# Plot individual test accuracy vs number of tasks as small dots
plt.scatter(
    [str(n_tasks) for n_tasks in results_df["Number of tasks"]],
    results_df["Global test accuracy"] * 100,
    label="Individual test accuracies",
    marker=".",
    c=results_df["Data seed"],
    cmap="winter",
    zorder=10,
    alpha=0.5,
)

# Plot average test accuracy over model seeds for each data seed vs. number of tasks as larger points
plt.scatter(
    [str(n_tasks) for n_tasks in avg_data_seeds["Number of tasks"]],
    avg_data_seeds["Global test accuracy"] * 100,
    label="Average over model seeds for each data seed",
    s=200,
    marker="_",
    c=avg_data_seeds["Data seed"],
    cmap="winter",
    alpha=0.5,
)

# Plot overall average test accuracy vs number of tasks as stars
plt.scatter(
    [str(n_tasks) for n_tasks in avg_n_tasks["Number of tasks"]],
    avg_n_tasks["Global test accuracy"] * 100,
    label="Average over all seeds for each number of tasks",
    s=200,
    marker="*",
    facecolor="none",
    edgecolor="firebrick",
    linewidths=1.3,
    # color="firebrick",
    zorder=20,
)

# Add labels and legend
plt.xlabel("Number of tasks", fontweight="bold")
plt.ylabel("Test accuracy / %", fontweight="bold")
data_set = data_set.replace("_", "")
plt.title(
    f"Robust seeds {data_set}: Global test accuracy vs. number of tasks",
    fontweight="bold",
)
plt.legend(loc="lower left", fontsize="small")
energy_str = f"Overall {(overall_energy / 1000):.2f} kWh consumed"
ax.text(
    0.75,
    0.95,
    energy_str,
    transform=ax.transAxes,
    fontsize="small",
    verticalalignment="top",
    fontweight="bold",
)
# plt.legend(loc='upper left', bbox_to_anchor=(0.5, 0.82), fontsize="small")
plt.savefig(pathlib.Path(root_dir) / f"{data_set}_acc_drop.png")

# Show the plot
plt.show()
