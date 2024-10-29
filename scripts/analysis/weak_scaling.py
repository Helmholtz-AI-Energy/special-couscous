import os
import pathlib
import re
import sys
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specialcouscous.utils.plot import (
    ERROR_KWARGS,
    GLOBAL_ERROR_KWARGS,
    GLOBAL_LINE_KWARGS,
    LINE_KWARGS,
)
from specialcouscous.utils.slurm import time_to_seconds

pd.set_option("display.max_rows", None)


def get_results_df(
    root_dir: Union[str, pathlib.Path],
) -> pd.DataFrame:
    """
    Construct results dataframe for plotting.

    Parameters
    ----------
    root_dir : Union[str, pathlib.Path]
        The path to the root directory containing the results.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the results.
    """
    data_set = str(root_dir).split(os.sep)[-1]
    # Dictionary to store the values grouped by (dataset, number_of_tasks, dataseed)
    results = defaultdict(list)

    # Loop over CSV files in root directory.
    for filename in pathlib.Path(root_dir).glob("**/*.csv"):
        # Extract relevant information from the directory structure
        print(f"Currently considered: {filename}")
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
        pattern_wall_clock_time = r"Job Wall-clock time: (\d+:\d+:\d+|\d+-\d+:\d+:\d+)"
        with open(filename, "r") as file:  # Load input text from the file.
            input_text = file.read()
        # Extract wall-clock time.
        wall_clock_time = time_to_seconds(
            re.search(pattern_wall_clock_time, input_text).group(1)  # type:ignore
        )
        print(f"Wall-clock time: {wall_clock_time} s")
        results[(data_set, number_of_tasks, model_seed)].append(wall_clock_time)
        # Extract energy consumed.
        pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
        energy_match = re.search(pattern_energy, input_text)
        energy_consumed = float(energy_match.group(0))  # type:ignore
        print(f"Energy consumed: {energy_consumed:.2f} Watthours")
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
    return results_df.sort_values(by=["Number of nodes", "Model seed"])


if __name__ == "__main__":
    # Get the root directory where results are stored from command line.
    root_dir = sys.argv[1]
    data_set = root_dir.split(os.sep)[-1]
    flavor = root_dir.split(os.sep)[-2].replace("_", " ")

    results_df = get_results_df(root_dir)

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
    std_acc_n_tasks = (
        results_df.groupby(["Number of nodes"])
        .agg({"Global test accuracy": "std"})
        .reset_index()
    )
    # For each parallelization level, get average of wall-clock time over model seeds.
    avg_time_n_tasks = (
        results_df.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "mean"})
        .reset_index()
    )
    std_time_n_tasks = (
        results_df.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "std"})
        .reset_index()
    )
    efficiency = (
        avg_time_n_tasks["Wall-clock time"][
            avg_time_n_tasks["Number of nodes"] == 1
        ].values
        / avg_time_n_tasks["Wall-clock time"]
    )

    std_efficiency = efficiency * np.sqrt(
        (
            std_time_n_tasks["Wall-clock time"][
                std_time_n_tasks["Number of nodes"] == 1
            ].values
            / avg_time_n_tasks["Wall-clock time"][
                avg_time_n_tasks["Number of nodes"] == 1
            ].values
        )
        ** 2
        + (std_time_n_tasks["Wall-clock time"] / avg_time_n_tasks["Wall-clock time"])
        ** 2
    )
    print(avg_time_n_tasks)
    print(avg_time_n_tasks["Wall-clock time"][avg_time_n_tasks["Number of nodes"] == 1])

    # Create the figure and the axes.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3, 5), sharex=True)
    # Settings
    labelsize = "small"
    legendsize = "xx-small"
    visible = False  # Whether to plot a grid or not

    # Set title
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Weak scaling {data_set}\n{flavor.capitalize()}",
        fontweight="bold",
        fontsize="small",
    )
    # Plot individual test accuracy vs. number of tasks.
    ax1.errorbar(
        [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
        avg_acc_n_tasks["Global test accuracy"] * 100,
        yerr=std_acc_n_tasks["Global test accuracy"] * 100,
        **GLOBAL_ERROR_KWARGS,
        label="Average global",
    )
    ax1.plot(
        [str(n_tasks) for n_tasks in avg_acc_n_tasks["Number of nodes"]],
        avg_acc_n_tasks["Global test accuracy"] * 100,
        **GLOBAL_LINE_KWARGS,
    )
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
    ax1.legend(loc="best", fontsize=legendsize)
    ax1.tick_params(axis="both", labelsize=labelsize)

    # Plot wall-clock time vs. number of tasks.
    ax2.errorbar(
        [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
        avg_time_n_tasks["Wall-clock time"] / 60,
        yerr=std_time_n_tasks["Wall-clock time"] / 60,
        **GLOBAL_ERROR_KWARGS,
    )
    ax2.plot(
        [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
        avg_time_n_tasks["Wall-clock time"] / 60,
        **GLOBAL_LINE_KWARGS,
    )
    ax2.set_ylabel("Runtime / min", fontweight="bold", fontsize=labelsize)
    ax2.grid(visible)
    ax2.tick_params(axis="both", labelsize=labelsize)
    ax2.set_ylim(0, 1.1 * avg_time_n_tasks["Wall-clock time"].max() / 60)

    # Plot overall average wall-clock time vs. number of tasks.
    ax3.errorbar(
        [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
        efficiency,
        yerr=std_efficiency,
        **GLOBAL_ERROR_KWARGS,
    )
    ax3.plot(
        [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
        efficiency,
        **GLOBAL_LINE_KWARGS,
    )
    ax3.plot(
        [str(n_tasks) for n_tasks in avg_time_n_tasks["Number of nodes"]],
        np.ones(len(avg_time_n_tasks["Number of nodes"])),
        label="Ideal",
        color="k",
        linestyle="dashed",
        linewidth=0.5,
    )
    ax3.set_ylabel("Efficiency", fontweight="bold", fontsize=labelsize)
    ax3.set_xlabel("Number of nodes", fontweight="bold", fontsize=labelsize)
    ax3.grid(visible)
    ax3.tick_params(axis="both", labelsize=labelsize)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc="lower right", fontsize=legendsize)
    plt.tight_layout()

    # Save the figure.
    flavor = flavor.replace(" ", "_")
    plt.savefig(pathlib.Path(root_dir) / f"{data_set}_{flavor}_weak_scaling.pdf")

    # Show the plot.
    plt.show()
