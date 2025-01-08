import os
import pathlib
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specialcouscous.utils.plot import (
    AVERAGE_KWARGS,
    ERROR_KWARGS,
    GLOBAL_ERROR_KWARGS,
    GLOBAL_LINE_KWARGS,
    INDIVIDUAL_KWARGS,
    LINE_KWARGS,
)
from specialcouscous.utils.slurm import time_to_seconds

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

if __name__ == "__main__":
    # Get the root directory where results are stored from command line.
    root_dir = sys.argv[1]
    data_set = root_dir.split(os.sep)[-1]
    # Dictionary to store the values grouped by (dataset, number_of_tasks, data seed)
    results = defaultdict(list)

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
            local_test_accuracy_mean = global_test_accuracy
            local_test_accuracy_std = 0
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
            pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
            energy_match = re.search(pattern_energy, input_text)
            energy_consumed = float(energy_match.group(0))  # type:ignore
            print(f"Energy Consumed: {energy_consumed:.2f} Watthours")
        else:
            wall_clock_time = None  # type:ignore
            energy_consumed = None  # type:ignore
        results[(data_set, number_of_tasks, model_seed)].append(wall_clock_time)
        results[(data_set, number_of_tasks, model_seed)].append(energy_consumed)

    # Save the results to a dataframe.
    results_df = pd.DataFrame(
        [(k[0], k[1], k[2], v[0], v[1], v[2], v[3], v[4]) for k, v in results.items()],
        columns=[
            "Dataset",
            "Number of nodes",
            "Model seed",
            "Local test accuracy",
            "Local test accuracy error",
            "Global test accuracy",
            "Wall-clock time",
            "Energy consumed",
        ],
    )
    results_df = results_df.sort_values(by=["Number of nodes", "Model seed"])
    print("Results dataframe:\n", results_df)

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

    std_global_acc = (
        results_df.groupby(["Number of nodes"])
        .agg(
            {
                "Global test accuracy": "std",
            }
        )
        .reset_index()
    )
    std_time = (
        results_df.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "std"})
        .reset_index()
    )

    # For each parallelization level, get average of wall-clock time over model seeds.
    time_n_tasks = (
        results_df.drop(["Local test accuracy", "Local test accuracy error"], axis=1)
        .dropna(axis=0)
        .reset_index()
        .drop("index", axis=1)
    )
    print("Wall-time dataframe\n:", time_n_tasks)
    efficiency = (
        time_n_tasks["Wall-clock time"][time_n_tasks["Number of nodes"] == 1].values
        / time_n_tasks["Wall-clock time"]
    )
    std_efficiency = efficiency * np.sqrt(
        (
            std_time["Wall-clock time"][std_time["Number of nodes"] == 1].values
            / time_n_tasks["Wall-clock time"][
                time_n_tasks["Number of nodes"] == 1
            ].values
        )
        ** 2
        + (std_time["Wall-clock time"] / time_n_tasks["Wall-clock time"]) ** 2
    )
    overall_energy = time_n_tasks["Energy consumed"].sum()
    print(f"Energy Consumed: {overall_energy:.2f} Watthours")
    # Create the figure and the axes.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3, 5))
    # Settings
    all_errors = True
    labelsize = "small"
    legendsize = "xx-small"
    visible = False  # Whether to plot a grid or not

    # Set title
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Weak scaling {data_set}",
        fontweight="bold",
        fontsize="small",
    )
    if all_errors:
        ax1.errorbar(
            [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
            avg_acc_n_tasks["Global test accuracy"] * 100,
            yerr=std_global_acc["Global test accuracy"] * 100,
            **GLOBAL_ERROR_KWARGS,
            label="Average global",
        )
        ax1.plot(
            [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
            avg_acc_n_tasks["Global test accuracy"] * 100,
            **GLOBAL_LINE_KWARGS,
        )
    else:
        # Plot individual test accuracy vs. number of tasks.
        ax1.scatter(
            [str(n_tasks) for n_tasks in results_df["Number of nodes"]],
            results_df["Global test accuracy"] * 100,
            label="Individual global",
            **INDIVIDUAL_KWARGS,
        )
        # Plot overall average test accuracy vs. number of tasks.
        ax1.scatter(
            [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
            avg_acc_n_tasks["Global test accuracy"] * 100,
            label="Average global",
            **AVERAGE_KWARGS,
        )
    # Plot average local test accuracy + error vs. number of tasks.
    ax1.errorbar(
        [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
        avg_acc_n_tasks["Local test accuracy"] * 100,
        yerr=avg_acc_n_tasks["Local test accuracy error"] * 100,
        **ERROR_KWARGS,
        label="Average local",
    )
    ax1.plot(
        [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
        avg_acc_n_tasks["Local test accuracy"] * 100,
        **LINE_KWARGS,
    )
    ax1.set_ylabel("Test accuracy / %", fontweight="bold", fontsize=labelsize)
    ax1.grid(visible)
    ax1.legend(fontsize=legendsize)
    ax1.tick_params(axis="both", labelsize=labelsize)

    # ----- WALL-CLOCK TIME vs. NUMBER OF NODES
    # NOTE: No error bars for runtimes and efficiency, as only one weak scaling run was conducted per parallelization
    # level for the sake of compute.
    ax2.scatter(
        [str(n_tasks) for n_tasks in results_df["Number of nodes"]],
        results_df["Wall-clock time"] / 60 / 60,
        **AVERAGE_KWARGS,
    )
    ax2.plot(
        [str(n_tasks) for n_tasks in time_n_tasks["Number of nodes"]],
        time_n_tasks["Wall-clock time"] / 60 / 60,
        **GLOBAL_LINE_KWARGS,
    )
    ax2.set_ylim(0, 1.1 * time_n_tasks["Wall-clock time"].max() / 60 / 60)
    ax2.tick_params(axis="both", labelsize=labelsize)
    ax2.set_ylabel("Runtime / h", fontweight="bold", fontsize=labelsize)
    ax2.grid(visible)
    energy_str = f"Overall {(overall_energy / 10**6):.2f} MWh consumed"

    ax2.text(
        0.05,
        0.125,
        energy_str,
        transform=ax2.transAxes,
        fontsize=legendsize,
        verticalalignment="top",
        fontweight="bold",
    )

    # Plot overall average efficiency vs. number of tasks.
    ax3.scatter(
        [str(n_tasks) for n_tasks in time_n_tasks["Number of nodes"]],
        efficiency,
        **AVERAGE_KWARGS,
    )
    ax3.plot(
        [str(n_tasks) for n_tasks in time_n_tasks["Number of nodes"]],
        efficiency,
        **GLOBAL_LINE_KWARGS,
    )
    ax3.plot(
        [str(n_tasks) for n_tasks in time_n_tasks["Number of nodes"]],
        np.ones(len(avg_acc_n_tasks["Number of nodes"])),
        label="Ideal",
        color="k",
        linestyle="dashed",
        linewidth=0.5,
    )
    ax3.set_ylabel("Efficiency", fontweight="bold", fontsize=labelsize)
    ax3.set_xlabel("Number of nodes", fontweight="bold", fontsize=labelsize)
    ax3.set_ylim(0, 1.1)
    ax3.grid(visible)
    ax3.tick_params(axis="both", labelsize=labelsize)
    ax3.legend(fontsize=legendsize)

    plt.tight_layout()

    # Save the figure.
    plt.savefig(pathlib.Path(root_dir) / f"{data_set}_weak_scaling.pdf")

    # Show the plot.
    plt.show()
