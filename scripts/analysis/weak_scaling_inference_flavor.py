import os
import pathlib
import re
import sys
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from specialcouscous.utils.slurm import time_to_seconds

pd.set_option("display.max_rows", None)

# Get the root directory where results are stored from command line.
root_dir = sys.argv[1]
data_set = root_dir.split(os.sep)[-1]
flavor = root_dir.split(os.sep)[-2].replace("_", " ")


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
    # Dictionary to store the values grouped by (dataset, number_of_tasks, dataseed)
    results = defaultdict(list)

    # Walk through the directory structure to find CSV files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                # Extract relevant information from the directory structure
                parts = dirpath.split(os.sep)
                number_of_tasks = int(parts[-2].split("_")[1])
                model_seed = int(
                    parts[-1].split("_")[2]
                )  # Extract model seed from path.
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
                model_seed = int(
                    parts[-1].split("_")[2]
                )  # Extract model seed from path.
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
    return results_df.sort_values(by=["Number of nodes", "Model seed"])


if __name__ == "__main__":
    # For each parallelization level, get average of test accuracy over model seeds.

    root_dir_no_shared_model = sys.argv[1]
    root_dir_shared_model = sys.argv[2]
    data_set = root_dir_no_shared_model.split(os.sep)[-1]

    results_df_no_shared_model = get_results_df(root_dir_no_shared_model)
    # energy_no_shared_model = results_df_no_shared_model["Energy consumed"].sum()
    results_df_shared_model = get_results_df(root_dir_shared_model)
    # energy_shared_model = results_df_shared_model["Energy consumed"].sum()

    avg_acc_n_tasks_no_shared_model = (
        results_df_no_shared_model.groupby(["Number of nodes"])
        .agg({"Global test accuracy": "mean"})
        .reset_index()
    )
    # For each parallelization level, get average of wall-clock time over model seeds.
    avg_time_n_tasks_no_shared_model = (
        results_df_no_shared_model.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "mean"})
        .reset_index()
    )

    avg_acc_n_tasks_shared_model = (
        results_df_shared_model.groupby(["Number of nodes"])
        .agg({"Global test accuracy": "mean"})
        .reset_index()
    )
    # For each parallelization level, get average of wall-clock time over model seeds.
    avg_time_n_tasks_shared_model = (
        results_df_shared_model.groupby(["Number of nodes"])
        .agg({"Wall-clock time": "mean"})
        .reset_index()
    )

    # Create the figure and the axes.
    fig, axes = plt.subplots(3, 2, figsize=(6, 6), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    # Set title.
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Weak scaling {data_set}",
        fontweight="bold",
    )
    ax1.set_title("No shared global model", fontsize="medium")

    # --- No shared model: Test accuracy ---
    # Plot individual test accuracy vs number of tasks (left y-axis)
    ax1.scatter(
        [str(n_tasks) for n_tasks in results_df_no_shared_model["Number of nodes"]],
        results_df_no_shared_model["Global test accuracy"] * 100,
        label="Individual test accuracies",
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Plot overall average test accuracy vs number of tasks as stars (left y-axis)
    ax1.scatter(
        [
            str(n_tasks)
            for n_tasks in avg_acc_n_tasks_no_shared_model["Number of nodes"]
        ],
        avg_acc_n_tasks_no_shared_model["Global test accuracy"] * 100,
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
    ax1.set_ylim(
        [
            0.999 * results_df_no_shared_model["Global test accuracy"].min() * 100,
            1.001 * results_df_no_shared_model["Global test accuracy"].max() * 100,
        ]
    )
    # ax1.legend(loc="lower right", fontsize="x-small")

    # --- Shared model: Test accuracy ---
    # Plot individual test accuracy vs number of tasks (left y-axis)
    ax2.set_title("Shared global model", fontsize="medium")
    ax2.scatter(
        [str(n_tasks) for n_tasks in results_df_shared_model["Number of nodes"]],
        results_df_shared_model["Global test accuracy"] * 100,
        label="Individual test accuracies",
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Plot overall average test accuracy vs number of tasks as stars (left y-axis)
    ax2.scatter(
        [str(n_tasks) for n_tasks in avg_acc_n_tasks_shared_model["Number of nodes"]],
        avg_acc_n_tasks_shared_model["Global test accuracy"] * 100,
        label="Average over all model seeds",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize="x-small")
    ax2.set_ylim(
        [
            0.999 * results_df_no_shared_model["Global test accuracy"].min() * 100,
            1.001 * results_df_no_shared_model["Global test accuracy"].max() * 100,
        ]
    )

    # --- No shared model: Wall-clock time ---
    # Plot wall-clock time vs number of tasks
    ax3.scatter(
        [str(n_tasks) for n_tasks in results_df_no_shared_model["Number of nodes"]],
        results_df_no_shared_model["Wall-clock time"]
        / 60,  # Assuming 'Wall-clock time' column exists
        label="Individual wall-clock times",
        marker=".",
        color="k",
        zorder=5,
        alpha=0.5,
    )
    ax3.scatter(
        [
            str(n_tasks)
            for n_tasks in avg_time_n_tasks_no_shared_model["Number of nodes"]
        ],
        avg_time_n_tasks_no_shared_model["Wall-clock time"] / 60,
        label="Average over all model seeds",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax3.set_ylabel("Wall-clock time / min", fontweight="bold")
    # ax3.legend(loc="upper left", fontsize="x-small")
    ax3.grid(True)
    ax3.set_ylim(
        [
            0.85 * results_df_no_shared_model["Wall-clock time"].min() / 60,
            1.05 * results_df_shared_model["Wall-clock time"].max() / 60,
        ]
    )

    # --- Shared model: Wall-clock time ---
    # Plot wall-clock time vs number of tasks (right y-axis)
    ax4.scatter(
        [str(n_tasks) for n_tasks in results_df_shared_model["Number of nodes"]],
        results_df_shared_model["Wall-clock time"]
        / 60,  # Assuming 'Wall-clock time' column exists
        label="Individual wall-clock times",
        marker=".",
        color="k",
        zorder=5,
        alpha=0.5,
    )
    ax4.scatter(
        [str(n_tasks) for n_tasks in avg_time_n_tasks_shared_model["Number of nodes"]],
        avg_time_n_tasks_shared_model["Wall-clock time"]
        / 60,  # Assuming 'Wall-clock time' column exists
        label="Average over all model seeds",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax4.legend(loc="upper left", fontsize="x-small")
    ax4.grid(True)
    ax4.set_ylim(
        [
            0.85 * results_df_no_shared_model["Wall-clock time"].min() / 60,
            1.05 * results_df_shared_model["Wall-clock time"].max() / 60,
        ]
    )

    # --- No shared model: Efficiency ---
    ax5.scatter(
        [
            str(n_tasks)
            for n_tasks in avg_time_n_tasks_no_shared_model["Number of nodes"]
        ],
        avg_time_n_tasks_no_shared_model["Wall-clock time"][
            avg_time_n_tasks_no_shared_model["Number of nodes"] == 1
        ].values
        / avg_time_n_tasks_no_shared_model["Wall-clock time"],
        label="Efficiency",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=15,
    )

    # Customize the right y-axis (wall-clock time)
    ax5.set_ylabel("Efficiency", fontweight="bold")
    ax5.set_xlabel("Number of nodes", fontweight="bold")
    ax5.grid(True)
    ax5.set_ylim(
        [
            0.80
            * (
                avg_time_n_tasks_shared_model["Wall-clock time"][
                    avg_time_n_tasks_shared_model["Number of nodes"] == 1
                ].values
                / avg_time_n_tasks_shared_model["Wall-clock time"]
            ).min(),
            1.10
            * (
                avg_time_n_tasks_shared_model["Wall-clock time"][
                    avg_time_n_tasks_shared_model["Number of nodes"] == 1
                ].values
                / avg_time_n_tasks_shared_model["Wall-clock time"]
            ).max(),
        ]
    )

    # --- Shared model: Efficiency ---
    ax6.scatter(
        [str(n_tasks) for n_tasks in avg_time_n_tasks_shared_model["Number of nodes"]],
        avg_time_n_tasks_shared_model["Wall-clock time"][
            avg_time_n_tasks_shared_model["Number of nodes"] == 1
        ].values
        / avg_time_n_tasks_shared_model["Wall-clock time"],
        label="Efficiency",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=15,
    )

    # Customize the right y-axis (wall-clock time)
    ax6.set_xlabel("Number of nodes", fontweight="bold")
    ax6.grid(True)
    ax6.set_ylim(
        [
            0.80
            * (
                avg_time_n_tasks_shared_model["Wall-clock time"][
                    avg_time_n_tasks_shared_model["Number of nodes"] == 1
                ].values
                / avg_time_n_tasks_shared_model["Wall-clock time"]
            ).min(),
            1.1
            * (
                avg_time_n_tasks_shared_model["Wall-clock time"][
                    avg_time_n_tasks_shared_model["Number of nodes"] == 1
                ].values
                / avg_time_n_tasks_shared_model["Wall-clock time"]
            ).max(),
        ]
    )
    # energy_str = f"Overall {(energy_shared_model/1000):.2f} kWh consumed"
    # ax6.text(
    #     0.05, 0.95, energy_str, transform=ax6.transAxes, fontsize="x-small", verticalalignment="top", fontweight="bold"
    # )
    # Ensure the layout is tight so labels don't overlap
    plt.tight_layout()
    #
    # Save the figure
    plt.savefig(pathlib.Path(root_dir_no_shared_model) / f"{data_set}_weak_scaling.png")

    # print(f"Overall energy consumed (no shared model): {(energy_no_shared_model/1000):.2f} kWh")
    # print(f"Overall energy consumed (shared model): {(energy_shared_model/1000):.2f} kWh")

    # Show the plot
    plt.show()
