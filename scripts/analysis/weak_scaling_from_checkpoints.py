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
    # Dictionary to store the values grouped by (dataset, number_of_tasks, data seed)
    results = defaultdict(list)
    target_columns = [
        "accuracy_global_test",
        "accuracy_local_test",
        "accuracy_test",
    ]  # Columns to extract from the CSV files

    # Walk through the directory structure to find CSV files
    for filename in pathlib.Path(root_dir).glob("**/*.csv"):
        print(f"Currently considered: {filename}")
        # Extract relevant information from the path.
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])
        model_seed = int(parts[-2].split("_")[2])  # Extract model seed from path.
        # Read the CSV file into a pandas dataframe.
        df = pd.read_csv(filename)

        # Extract the value from the target column and store it
        if "accuracy_global_test" in df.columns:  # Parallel runs
            global_test_accuracy = df.loc[
                df["comm_rank"] == "global", "accuracy_global_test"
            ].values[0]
            local_test_accuracy_mean = df["accuracy_local_test"].dropna().mean()
            local_test_accuracy_std = df["accuracy_local_test"].dropna().std()
        elif "accuracy_test" in df.columns:  # Serial runs
            global_test_accuracy = df["accuracy_test"].values[0]
            local_test_accuracy_mean = None
            local_test_accuracy_std = None
        else:
            raise ValueError("No valid test accuracy column in dataframe!")

        print(
            f"{data_set}: {number_of_tasks} tasks, model seed {model_seed}: Global test acc.: {global_test_accuracy}"
        )
        results[(data_set, number_of_tasks, model_seed)].append(
            local_test_accuracy_mean
        )
        results[(data_set, number_of_tasks, model_seed)].append(local_test_accuracy_std)
        results[(data_set, number_of_tasks, model_seed)].append(global_test_accuracy)

    for filename in pathlib.Path(root_dir).glob("**/*.out"):
        print(f"Currently considered: {filename}")
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])
        model_seed = int(parts[-2].split("_")[2])  # Extract model seed from path.
        if model_seed == 1:
            pattern_wall_clock_time = (
                r"Job Wall-clock time: (\d+:\d+:\d+|\d+-\d+:\d+:\d+)"
            )
            with open(filename, "r") as file:  # Load input text from the file.
                input_text = file.read()
            # Extract wall-clock time.
            wall_clock_time = time_to_seconds(
                re.search(pattern_wall_clock_time, input_text).group(1)  # type:ignore
            )
            print(f"Wall-clock time: {wall_clock_time} s")
        else:
            wall_clock_time = None
        results[(data_set, number_of_tasks, model_seed)].append(wall_clock_time)

    # Save the results to a dataframe.
    results_df = pd.DataFrame(
        [(k[0], k[1], k[2], v[0], v[1], v[2], v[3]) for k, v in results.items()],
        columns=[
            "Dataset",
            "Number of nodes",
            "Model seed",
            "Local test accuracy",
            "Local test accuracy error",
            "Global test accuracy",
            "Wall-clock time",
        ],
    )
    results_df = results_df.sort_values(by=["Number of nodes", "Model seed"])
    print(results_df)

    # For each parallelization level, get average of test accuracy over model seeds.
    avg_acc_n_tasks = (
        results_df.groupby(["Number of nodes"])
        .agg(
            {
                "Global test accuracy": "mean",
                "Local test accuracy": "mean",
                "Local test accuracy error": "mean",
            }
        )
        .reset_index()
    )
    # For each parallelization level, get average of wall-clock time over model seeds.
    time_n_tasks = results_df.dropna(axis=0)

    # print(avg_time_n_tasks["Wall-clock time"][avg_time_n_tasks["Number of nodes"] == 1])

    # Create the figure and the axes.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3, 5), sharex=True)
    # Settings
    labelsize = "small"
    legendsize = "xx-small"
    visible = False  # Whether to plot a grid or not
    average_kwargs = {
        "marker": "x",
        "s": 38,
        "color": "C0",
        "linewidths": 1,
    }
    individual_kwargs = {
        "marker": ",",
        "color": "k",
        "s": 5,
        "alpha": 0.3,
    }
    error_kwargs = {
        "fmt": "_",
        "color": "C2",
        "ecolor": "C2",
        "elinewidth": 1,
        "capsize": 0.9,
        "ms": 5,
    }
    # Set title
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Weak scaling {data_set}",
        fontweight="bold",
        fontsize="small",
    )
    # Plot individual test accuracy vs. number of tasks.
    ax1.scatter(
        [str(n_tasks) for n_tasks in results_df["Number of nodes"]],
        results_df["Global test accuracy"] * 100,
        label="Individual global",
        **individual_kwargs,
        zorder=20,
    )
    # Plot overall average test accuracy vs. number of tasks.
    ax1.scatter(
        [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
        avg_acc_n_tasks["Global test accuracy"] * 100,
        label="Average global",
        **average_kwargs,
        zorder=10,
    )
    # Plot average local test accuracy + error vs. number of tasks.
    ax1.errorbar(
        [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
        avg_acc_n_tasks["Local test accuracy"] * 100,
        yerr=avg_acc_n_tasks["Local test accuracy error"]
        * 100,  # Plot error bars 100 x magnified?
        **error_kwargs,
        zorder=20,
        label="Average local",
    )
    ax1.set_ylabel("Test accuracy / %", fontweight="bold", fontsize=labelsize)
    ax1.grid(visible)
    ax1.legend(fontsize=legendsize)
    ax1.tick_params(axis="both", labelsize=labelsize)

    # Plot wall-clock time vs. number of tasks.
    ax2.scatter(
        [str(n_tasks) for n_tasks in results_df["Number of nodes"]],
        results_df["Wall-clock time"] / 60 / 60,
        label="Individual",
        **individual_kwargs,
        zorder=10,
    )
    # ax2.scatter(
    #     [str(n_tasks) for n_tasks in time_n_tasks["Number of nodes"]],
    #     time_n_tasks["Wall-clock time"] / 60,
    #     label="Average over all model seeds",
    #     s=80,
    #     marker="X",
    #     facecolor="none",
    #     edgecolor="k",
    #     linewidths=1.3,
    #     zorder=20,
    # )
    ax2.set_ylim(0, 1.1 * results_df["Wall-clock time"].max() / 60 / 60)
    ax2.set_ylabel("Wall-clock time / h", fontweight="bold", fontsize=labelsize)
    # ax2.legend(loc="lower right", fontsize=legendsize)
    ax2.grid(visible)
    ax2.tick_params(axis="both", labelsize=labelsize)

    # print(
    #     time_n_tasks["Wall-clock time"][
    #         time_n_tasks["Number of nodes"] == 1
    #     ].values
    #     / time_n_tasks["Wall-clock time"]
    # )
    # Plot overall average wall-clock time vs. number of tasks.
    # ax3.scatter(
    #     [str(n_tasks) for n_tasks in time_n_tasks["Number of nodes"]],
    #     time_n_tasks["Wall-clock time"][
    #         time_n_tasks["Number of nodes"] == 1
    #     ].values
    #     / time_n_tasks["Wall-clock time"],
    #     label="Efficiency",
    #     s=80,
    #     marker="X",
    #     facecolor="none",
    #     edgecolor="k",
    #     linewidths=1.3,
    #     zorder=15,
    # )
    ax3.set_ylabel("Efficiency", fontweight="bold", fontsize=labelsize)
    ax3.set_xlabel("Number of nodes", fontweight="bold", fontsize=labelsize)
    ax3.grid(visible)
    ax3.tick_params(axis="both", labelsize=labelsize)

    plt.tight_layout()

    # Save the figure.
    # flavor = flavor.replace(" ", "_")
    plt.savefig(pathlib.Path(root_dir) / f"{data_set}_weak_scaling.pdf")

    # Show the plot.
    plt.show()
