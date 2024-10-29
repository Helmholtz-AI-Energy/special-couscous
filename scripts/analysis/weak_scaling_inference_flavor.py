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
    # For each parallelization level, get average of test accuracy over model seeds.

    root_dir_no_shared_model = sys.argv[1]
    root_dir_shared_model = sys.argv[2]
    data_set = root_dir_no_shared_model.split(os.sep)[-1]

    results_df_no_shared_model = get_results_df(root_dir_no_shared_model)
    results_df_shared_model = get_results_df(root_dir_shared_model)

    print(results_df_no_shared_model)
    print(results_df_shared_model)

    # --- NO SHARED MODEL ---
    # Average accuracies + error
    avg_acc_no_shared_model = (
        results_df_no_shared_model.groupby(["Number of nodes"])
        .agg(
            {
                "Global test accuracy": "mean",
                "Local test accuracy": "mean",
                "Local test accuracy error": "mean",
            }
        )
        .reset_index()
    )
    std_acc_no_shared_model = (
        results_df_no_shared_model.groupby(["Number of nodes"])
        .agg({"Global test accuracy": "std"})
        .reset_index()
    )
    # Average wall-clock times + error
    avg_time_no_shared_model = (
        results_df_no_shared_model.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "mean"})
        .reset_index()
    )
    std_time_no_shared_model = (
        results_df_no_shared_model.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "std"})
        .reset_index()
    )
    # Efficiency + error
    efficiency_no_shared_model = (
        avg_time_no_shared_model["Wall-clock time"][
            avg_time_no_shared_model["Number of nodes"] == 1
        ].values
        / avg_time_no_shared_model["Wall-clock time"]
    )

    std_efficiency_no_shared_model = efficiency_no_shared_model * np.sqrt(
        (
            std_time_no_shared_model["Wall-clock time"][
                std_time_no_shared_model["Number of nodes"] == 1
            ].values
            / avg_time_no_shared_model["Wall-clock time"][
                avg_time_no_shared_model["Number of nodes"] == 1
            ].values
        )
        ** 2
        + (
            std_time_no_shared_model["Wall-clock time"]
            / avg_time_no_shared_model["Wall-clock time"]
        )
        ** 2
    )
    # --- SHARED MODEL---
    # Average accuracies + error
    avg_acc_shared_model = (
        results_df_shared_model.groupby(["Number of nodes"])
        .agg(
            {
                "Global test accuracy": "mean",
                "Local test accuracy": "mean",
                "Local test accuracy error": "mean",
            }
        )
        .reset_index()
    )
    std_acc_n_shared_model = (
        results_df_shared_model.groupby(["Number of nodes"])
        .agg({"Global test accuracy": "std"})
        .reset_index()
    )
    # Average wall-clock times + error
    avg_time_shared_model = (
        results_df_shared_model.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "mean"})
        .reset_index()
    )
    std_time_n_tasks_shared_model = (
        results_df_shared_model.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "std"})
        .reset_index()
    )
    # Efficiency + error
    efficiency_shared_model = (
        avg_time_shared_model["Wall-clock time"][
            avg_time_shared_model["Number of nodes"] == 1
        ].values
        / avg_time_shared_model["Wall-clock time"]
    )

    std_efficiency_shared_model = efficiency_shared_model * np.sqrt(
        (
            std_time_n_tasks_shared_model["Wall-clock time"][
                std_time_n_tasks_shared_model["Number of nodes"] == 1
            ].values
            / avg_time_shared_model["Wall-clock time"][
                avg_time_shared_model["Number of nodes"] == 1
            ].values
        )
        ** 2
        + (
            std_time_n_tasks_shared_model["Wall-clock time"]
            / avg_time_shared_model["Wall-clock time"]
        )
        ** 2
    )

    # Create the figure and the axes.
    fig, axes = plt.subplots(3, 2, figsize=(5, 5), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    # Settings
    labelsize = "small"
    legendsize = "xx-small"
    visible = False  # Whether to plot a grid or not

    # Set title.
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Weak scaling {data_set}",
        fontweight="bold",
        fontsize="medium",
    )
    ax1.set_title("No shared global model", fontsize="small")

    # --- No shared model: Test accuracy ---
    # Plot individual test accuracy vs number of tasks (left y-axis)
    ax1.errorbar(
        [str(n_tasks) for n_tasks in avg_acc_no_shared_model["Number of nodes"]],
        avg_acc_no_shared_model["Global test accuracy"] * 100,
        yerr=std_acc_no_shared_model["Global test accuracy"] * 100,
        **GLOBAL_ERROR_KWARGS,
        label="Average global",
    )
    ax1.plot(
        [str(n_tasks) for n_tasks in avg_acc_no_shared_model["Number of nodes"]],
        avg_acc_no_shared_model["Global test accuracy"] * 100,
        **GLOBAL_LINE_KWARGS,
    )
    ax1.errorbar(
        [str(n_tasks) for n_tasks in avg_acc_no_shared_model["Number of nodes"]],
        avg_acc_no_shared_model["Local test accuracy"] * 100,
        yerr=avg_acc_no_shared_model["Local test accuracy error"] * 100,
        **ERROR_KWARGS,
        label="Average local",
    )
    ax1.plot(
        [str(n_tasks) for n_tasks in avg_acc_no_shared_model["Number of nodes"]],
        avg_acc_no_shared_model["Local test accuracy"] * 100,
        **LINE_KWARGS,
    )
    # Customize the left y-axis (test accuracy)
    ax1.set_ylabel("Test accuracy / %", fontweight="bold", fontsize=labelsize)
    ax1.grid(visible)
    ax1.tick_params(axis="both", labelsize=labelsize)
    ax1.set_ylim(
        [
            0.99 * results_df_no_shared_model["Global test accuracy"].min() * 100,
            1.01 * results_df_no_shared_model["Global test accuracy"].max() * 100,
        ]
    )
    ax1.legend(fontsize=legendsize, loc="best")

    # --- Shared model: Test accuracy ---
    # Plot individual test accuracy vs number of tasks (left y-axis)
    ax2.set_title("Shared global model", fontsize="small")
    ax2.errorbar(
        [str(n_tasks) for n_tasks in avg_acc_shared_model["Number of nodes"]],
        avg_acc_shared_model["Global test accuracy"] * 100,
        yerr=std_acc_n_shared_model["Global test accuracy"] * 100,
        **GLOBAL_ERROR_KWARGS,
        label="Average",
    )
    # Plot overall average test accuracy vs number of tasks as stars (left y-axis)
    ax2.plot(
        [str(n_tasks) for n_tasks in avg_acc_shared_model["Number of nodes"]],
        avg_acc_shared_model["Global test accuracy"] * 100,
        **GLOBAL_LINE_KWARGS,
    )
    ax2.errorbar(
        [str(n_tasks) for n_tasks in avg_acc_shared_model["Number of nodes"]],
        avg_acc_shared_model["Local test accuracy"] * 100,
        yerr=avg_acc_shared_model["Local test accuracy error"] * 100,
        **ERROR_KWARGS,
        label="Average local",
    )
    ax2.plot(
        [str(n_tasks) for n_tasks in avg_acc_shared_model["Number of nodes"]],
        avg_acc_shared_model["Local test accuracy"] * 100,
        **LINE_KWARGS,
    )
    ax2.grid(visible)
    ax2.tick_params(axis="both", labelsize=labelsize)
    ax2.set_ylim(
        [
            0.99 * results_df_no_shared_model["Global test accuracy"].min() * 100,
            1.01 * results_df_no_shared_model["Global test accuracy"].max() * 100,
        ]
    )

    # --- No shared model: Wall-clock time ---
    # Plot wall-clock time vs number of tasks
    ax3.errorbar(
        [str(n_tasks) for n_tasks in avg_time_no_shared_model["Number of nodes"]],
        avg_time_no_shared_model["Wall-clock time"] / 60,
        yerr=std_time_no_shared_model["Wall-clock time"] / 60,
        **GLOBAL_ERROR_KWARGS,
        label="Average",
    )
    ax3.plot(
        [str(n_tasks) for n_tasks in avg_time_no_shared_model["Number of nodes"]],
        avg_time_no_shared_model["Wall-clock time"] / 60,
        **GLOBAL_LINE_KWARGS,
    )
    ax3.set_ylabel("Runtime / min", fontweight="bold", fontsize=labelsize)
    ax3.grid(visible)
    ax3.set_ylim(
        [
            0.85 * results_df_no_shared_model["Wall-clock time"].min() / 60,
            1.05 * results_df_shared_model["Wall-clock time"].max() / 60,
        ]
    )
    ax3.tick_params(axis="both", labelsize=labelsize)
    ax3.legend(fontsize=legendsize, loc="upper right")

    # --- Shared model: Wall-clock time ---
    # Plot wall-clock time vs number of tasks (right y-axis)
    ax4.errorbar(
        [str(n_tasks) for n_tasks in avg_time_shared_model["Number of nodes"]],
        avg_time_shared_model["Wall-clock time"] / 60,
        yerr=std_time_n_tasks_shared_model["Wall-clock time"] / 60,
        **GLOBAL_ERROR_KWARGS,
        label="Average",
    )
    ax4.plot(
        [str(n_tasks) for n_tasks in avg_time_shared_model["Number of nodes"]],
        avg_time_shared_model["Wall-clock time"] / 60,
        **GLOBAL_LINE_KWARGS,
    )
    ax4.tick_params(axis="both", labelsize=labelsize)
    ax4.grid(visible)
    ax4.set_ylim(
        [
            0.85 * results_df_no_shared_model["Wall-clock time"].min() / 60,
            1.05 * results_df_shared_model["Wall-clock time"].max() / 60,
        ]
    )

    # --- No shared model: Efficiency ---
    ax5.errorbar(
        [str(n_tasks) for n_tasks in avg_time_no_shared_model["Number of nodes"]],
        efficiency_no_shared_model,
        yerr=std_efficiency_no_shared_model,
        label="Average",
        **GLOBAL_ERROR_KWARGS,
    )
    ax5.plot(
        [str(n_tasks) for n_tasks in avg_time_no_shared_model["Number of nodes"]],
        efficiency_no_shared_model,
        **GLOBAL_LINE_KWARGS,
    )
    ax5.plot(
        [str(n_tasks) for n_tasks in avg_time_shared_model["Number of nodes"]],
        np.ones(len(avg_time_shared_model["Number of nodes"])),
        label="Ideal",
        color="k",
        linestyle="dashed",
        linewidth=0.5,
    )
    ax5.set_ylabel("Efficiency", fontweight="bold", fontsize=labelsize)
    ax5.set_xlabel("Number of nodes", fontweight="bold", fontsize=labelsize)
    ax5.grid(visible)
    ax5.set_ylim([0.0, 1.1])
    ax5.tick_params(axis="both", labelsize=labelsize)
    ax5.legend(fontsize=legendsize, loc="lower right")

    # --- Shared model: Efficiency ---
    ax6.errorbar(
        [str(n_tasks) for n_tasks in avg_time_shared_model["Number of nodes"]],
        efficiency_shared_model,
        yerr=std_efficiency_shared_model,
        label="Efficiency",
        **GLOBAL_ERROR_KWARGS,
    )
    ax6.plot(
        [str(n_tasks) for n_tasks in avg_time_shared_model["Number of nodes"]],
        efficiency_shared_model,
        **GLOBAL_LINE_KWARGS,
    )
    ax6.plot(
        [str(n_tasks) for n_tasks in avg_time_shared_model["Number of nodes"]],
        np.ones(len(avg_time_shared_model["Number of nodes"])),
        label="Ideal",
        color="k",
        linestyle="dashed",
        linewidth=0.5,
    )
    ax6.set_xlabel("Number of nodes", fontweight="bold", fontsize=labelsize)
    ax6.grid(visible)
    ax6.set_ylim([0.0, 1.1])
    ax6.tick_params(axis="both", labelsize=labelsize)
    plt.tight_layout()
    plt.savefig(pathlib.Path(root_dir_no_shared_model) / f"{data_set}_weak_scaling.pdf")
    plt.show()
