import os
import pathlib
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from specialcouscous.utils.slurm import time_to_seconds

pd.set_option("display.max_rows", None)


if __name__ == "__main__":
    # Get the root directory where results are stored from command line.
    root_dir = sys.argv[1]
    data_set = root_dir.split(os.sep)[-1]
    flavor = root_dir.split(os.sep)[-2].replace("_", " ")
    # Dictionary to store the values grouped by (dataset, number_of_tasks, data seed)
    results = defaultdict(list)
    target_columns = [
        "accuracy_global_test",
        "accuracy_test",
    ]  # Columns to extract from the CSV files

    # Walk through the directory structure to find CSV files
    for filename in pathlib.Path(root_dir).glob("**/*.csv"):
        print(f"Currently considered: {filename}")
        # Extract relevant information from the path.
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])
        model_seed = int(parts[-2].split("_")[2])  # Extract model seed from path.
        print(f"{data_set}: {number_of_tasks} tasks, model seed {model_seed}")
        # Read the CSV file into a pandas dataframe.
        df = pd.read_csv(filename)

        # Extract the value from the target column and store it
        if "accuracy_global_test" in df.columns:  # Parallel runs
            global_test_accuracy = df.loc[
                df["comm_rank"] == "global", "accuracy_global_test"
            ].values[0]
        elif "accuracy_test" in df.columns:  # Serial runs
            global_test_accuracy = df["accuracy_test"].values[0]
        else:
            raise ValueError("No valid test accuracy column in dataframe!")

        print(f"Global test accuracy: {global_test_accuracy}")
        results[(data_set, number_of_tasks, model_seed)].append(global_test_accuracy)

    for filename in pathlib.Path(root_dir).glob("**/*.out"):
        print(f"Currently considered: {filename}")
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])
        model_seed = int(parts[-2].split("_")[2])  # Extract model seed from path.
        pattern_wall_clock_time = r"Job Wall-clock time: (\d+:\d+:\d+|\d+-\d+:\d+:\d+)"
        with open(filename, "r") as file:  # Load input text from the file.
            input_text = file.read()
        # Extract wall-clock time.
        wall_clock_time = time_to_seconds(
            re.search(pattern_wall_clock_time, input_text).group(1)  # type:ignore
        )
        print(f"Wall-clock time: {wall_clock_time} s")
        results[(data_set, number_of_tasks, model_seed)].append(wall_clock_time)

    # Save the results to a dataframe.
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

    # Create the figure and the axes.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
    # Set title
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Weak scaling {data_set} {flavor}",
        fontweight="bold",
    )
    # Plot individual test accuracy vs. number of tasks.
    ax1.scatter(
        [str(n_tasks) for n_tasks in results_df["Number of nodes"]],
        results_df["Global test accuracy"] * 100,
        label="Individual test accuracies",
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Plot overall average test accuracy vs. number of tasks as stars.
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
    ax1.set_ylabel("Test accuracy / %", fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize="small")

    # Plot wall-clock time vs. number of tasks.
    ax2.scatter(
        [str(n_tasks) for n_tasks in results_df["Number of nodes"]],
        results_df["Wall-clock time"] / 60,
        label="Individual wall-clock times",
        marker=".",
        color="k",
        zorder=5,
        alpha=0.5,
    )
    ax2.scatter(
        [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
        avg_time_n_tasks["Wall-clock time"] / 60,
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
        avg_time_n_tasks["Wall-clock time"][
            avg_time_n_tasks["Number of nodes"] == 1
        ].values
        / avg_time_n_tasks["Wall-clock time"]
    )
    # Plot overall average wall-clock time vs. number of tasks.
    ax3.scatter(
        [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
        avg_time_n_tasks["Wall-clock time"][
            avg_time_n_tasks["Number of nodes"] == 1
        ].values
        / avg_time_n_tasks["Wall-clock time"],
        label="Efficiency",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=15,
    )
    ax3.set_ylabel("Efficiency", fontweight="bold")
    ax3.set_xlabel("Number of nodes", fontweight="bold")
    ax3.grid(True)

    plt.tight_layout()

    # Save the figure.
    flavor = flavor.replace(" ", "_")
    plt.savefig(pathlib.Path(root_dir) / f"{data_set}_{flavor}_weak_scaling.png")

    # Show the plot.
    plt.show()
