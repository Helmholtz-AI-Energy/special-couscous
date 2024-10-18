import os
import pathlib
import re
import sys
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)


def convert_to_gb(memory_value: str, unit: str) -> float:
    """
    Convert memory value to GB.

    Parameters
    ----------
    memory_value : str
        The memory value.
    unit : str
        The unit of the value. Must be either "MB", "GB", or "TB".
    """
    value = float(memory_value)  # Convert the captured value to float
    if unit == "TB":
        return value * 1024  # Convert TB to GB
    elif unit == "MB":
        return value / 1024  # Convert MB to GB
    return value  # GB remains the same


def get_results_df(
    path_to_root: Union[str, pathlib.Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct results dataframe for plotting.

    Parameters
    ----------
    path_to_root : Union[str, pathlib.Path]
        The path to the root directory containing the results.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the results.
    pd.DataFrame
        The dataframe containing the results averaged over model seeds for each number of nodes.
    """
    # Dictionary to store the values grouped by (dataset, number_of_tasks, dataseed)
    results = defaultdict(list)

    # The specific column you want to average in the CSV files
    target_columns = [
        "time_sec_data_generation",
        "time_sec_forest_creation",
        "time_sec_training",
        "time_sec_all-gathering_model",
        "time_sec_test",
    ]

    # Walk through the directory structure to find CSV files
    for dirpath, dirnames, filenames in os.walk(path_to_root):
        for filename in filenames:
            if filename.endswith(".csv"):
                # Extract relevant information from the directory structure
                parts = dirpath.split(os.sep)
                number_of_tasks = int(parts[-2].split("_")[1])
                model_seed = int(
                    parts[-1].split("_")[2]
                )  # Extract model seed from path.
                # Read the CSV file into a pandas dataframe.
                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path)

                # Extract the value from the target column and store it
                # Parallel runs:
                for column in target_columns:
                    if column in df.columns:
                        result = df.loc[df["comm_rank"] == "global", column].values[0]
                    else:
                        result = np.nan
                    results[(data_set, number_of_tasks, model_seed)].append(result)

    for dirpath, dirnames, filenames in os.walk(path_to_root):
        for filename in filenames:
            if filename.endswith(".out"):
                parts = dirpath.split(os.sep)
                number_of_tasks = int(parts[-2].split("_")[1])
                model_seed = int(
                    parts[-1].split("_")[2]
                )  # Extract model seed from path.
                pattern_memory = r"Memory Utilized:\s*([0-9]+\.?[0-9]*)\s*(MB|GB|TB)"
                pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
                with open(
                    os.path.join(dirpath, filename), "r"
                ) as file:  # Load input text from the file.
                    input_text = file.read()
                    print(dirpath)
                memory_match = re.search(pattern_memory, input_text)
                energy_match = re.search(pattern_energy, input_text)
                memory_utilized = memory_match.group(1)  # type:ignore
                energy_consumed = float(energy_match.group(0))  # type:ignore
                unit = memory_match.group(2)  # type:ignore
                memory_in_gb = convert_to_gb(memory_utilized, unit)
                print(
                    f"Memory Utilized: {memory_in_gb:.2f} GB\nEnergy Consumed: {energy_consumed:.2f} Watthours"
                )
                results[(data_set, number_of_tasks, model_seed)].append(memory_in_gb)
                results[(data_set, number_of_tasks, model_seed)].append(energy_consumed)

    # Save the results to a pandas dataframe.
    results_df = pd.DataFrame(
        [
            (k[0], k[1], k[2], v[0], v[1], v[2], v[3], v[4], v[5], v[6])
            for k, v in results.items()
        ],
        columns=[
            "Dataset",
            "Number of nodes",
            "Model seed",
            "Time for data generation",
            "Time for forest creation",
            "Time for training",
            "Time for all-gathering model",
            "Time for evaluation",
            "Memory used",
            "Energy consumed",
        ],
    )
    results_df = results_df.sort_values(by=["Number of nodes", "Model seed"])
    results_df = results_df[results_df["Number of nodes"] != 1]
    results_df["Overall time"] = (
        results_df.drop(columns=["Dataset", "Number of nodes", "Model seed"])
        .fillna(0)
        .sum(axis=1)
    )
    print(results_df)

    # For each parallelization level, get average of test accuracy over model seeds.
    avg_n_tasks = (
        results_df.groupby(["Number of nodes"])[
            [
                "Time for data generation",
                "Time for forest creation",
                "Time for training",
                "Time for all-gathering model",
                "Time for evaluation",
                "Overall time",
                "Memory used",
                "Energy consumed",
            ]
        ]
        .mean()
        .reset_index()
    )
    print(avg_n_tasks)
    return results_df, avg_n_tasks


if __name__ == "__main__":
    # Get the root directory where results are stored from command line.
    root_dir_no_shared_model = sys.argv[1]
    root_dir_shared_model = sys.argv[2]
    data_set = root_dir_no_shared_model.split(os.sep)[-1]

    results_df_no_shared_model, avg_n_tasks_no_shared_model = get_results_df(
        root_dir_no_shared_model
    )
    energy_no_shared_model = results_df_no_shared_model["Energy consumed"].sum()
    results_df_shared_model, avg_n_tasks_shared_model = get_results_df(
        root_dir_shared_model
    )
    energy_shared_model = results_df_shared_model["Energy consumed"].sum()
    # Create the figure and the first axis for test accuracy
    fig, axes = plt.subplots(3, 2, figsize=(6, 6), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    # Set title
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Inference flavor comparison {data_set}",
        fontweight="bold",
    )
    ax1.set_title("No shared global model", fontsize="medium")
    # Plot times over number of nodes for no shared global model.
    # Evaluate
    # Individual data points
    ax1.scatter(
        [str(n_tasks) for n_tasks in results_df_no_shared_model["Number of nodes"]],
        results_df_no_shared_model["Time for evaluation"] / 60,
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Average
    ax1.scatter(
        [str(n_tasks) for n_tasks in avg_n_tasks_no_shared_model["Number of nodes"]],
        avg_n_tasks_no_shared_model["Time for evaluation"] / 60,
        label="Evaluate model",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax1.set_ylim(
        [
            0.6 * results_df_shared_model["Time for all-gathering model"].min() / 60,
            1.5 * results_df_shared_model["Time for evaluation"].max() / 60,
        ]
    )
    # Overall time
    # Individual data points
    # ax1.scatter(
    #     [str(n_tasks) for n_tasks in results_df_no_shared_model["Number of nodes"]],
    #     results_df_no_shared_model["Overall time"] / 60,
    #     label="Evaluate model",
    #     marker=".",
    #     color="red",
    #     zorder=10,
    #     alpha=0.5,
    # )
    # # Average
    # ax1.scatter(
    #     [
    #         str(n_tasks)
    #         for n_tasks in avg_times_n_tasks_no_shared_model["Number of nodes"]
    #     ],
    #     avg_times_n_tasks_no_shared_model["Overall time"] / 60,
    #     label="Overall time",
    #     s=80,
    #     marker="X",
    #     facecolor="none",
    #     edgecolor="red",
    #     linewidths=1.3,
    #     zorder=20,
    # )
    ax1.set_yscale("log", base=2)
    ax1.set_ylabel("Time / min", fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize="x-small")

    # Plot times over number of nodes for shared global model.
    # All-gather global shared model.
    ax2.set_title("Shared global model", fontsize="medium")
    # Individual data points
    ax2.scatter(
        [str(n_tasks) for n_tasks in results_df_shared_model["Number of nodes"]],
        results_df_shared_model["Time for all-gathering model"] / 60,
        marker=".",
        color="blue",
        zorder=100,
        alpha=0.5,
    )
    print(avg_n_tasks_shared_model["Time for all-gathering model"])
    # Average
    ax2.scatter(
        [str(n_tasks) for n_tasks in avg_n_tasks_shared_model["Number of nodes"]],
        avg_n_tasks_shared_model["Time for all-gathering model"] / 60,
        label="All-gather model",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="blue",
        linewidths=1.3,
        zorder=200,
    )
    # Evaluate
    # Individual data points
    ax2.scatter(
        [str(n_tasks) for n_tasks in results_df_shared_model["Number of nodes"]],
        results_df_shared_model["Time for evaluation"] / 60,
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Average
    ax2.scatter(
        [str(n_tasks) for n_tasks in avg_n_tasks_shared_model["Number of nodes"]],
        avg_n_tasks_shared_model["Time for evaluation"] / 60,
        label="Evaluate model",
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax2.set_ylim(
        [
            0.6 * results_df_shared_model["Time for all-gathering model"].min() / 60,
            1.5 * results_df_shared_model["Time for evaluation"].max() / 60,
        ]
    )
    ax2.set_yscale("log", base=2)

    # Overall time
    # Individual data points
    # ax2.scatter(
    #     [str(n_tasks) for n_tasks in results_df_shared_model["Number of nodes"]],
    #     results_df_shared_model["Overall time"] / 60,
    #     label="Evaluate model",
    #     marker=".",
    #     color="red",
    #     zorder=10,
    #     alpha=0.5,
    # )
    # # Average
    # ax2.scatter(
    #     [str(n_tasks) for n_tasks in avg_times_n_tasks_shared_model["Number of nodes"]],
    #     avg_times_n_tasks_shared_model["Overall time"] / 60,
    #     label="Overall time",
    #     s=80,
    #     marker="X",
    #     facecolor="none",
    #     edgecolor="red",
    #     linewidths=1.3,
    #     zorder=20,
    # )
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize="x-small")

    # Memory used
    # No shared global model
    ax3.scatter(
        [str(n_tasks) for n_tasks in results_df_no_shared_model["Number of nodes"]],
        results_df_no_shared_model["Memory used"],
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Average
    ax3.scatter(
        [str(n_tasks) for n_tasks in avg_n_tasks_no_shared_model["Number of nodes"]],
        avg_n_tasks_no_shared_model["Memory used"],
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax3.set_ylim(
        [
            0.6 * results_df_no_shared_model["Memory used"].min(),
            1.6 * results_df_shared_model["Memory used"].max(),
        ]
    )
    ax3.set_yscale("log", base=2)
    ax3.set_ylabel("Memory / GB", fontweight="bold")
    # ax3.set_xlabel("Number of nodes", fontweight="bold")
    ax3.grid(True)

    # No shared global model
    ax4.scatter(
        [str(n_tasks) for n_tasks in results_df_shared_model["Number of nodes"]],
        results_df_shared_model["Memory used"],
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Average
    ax4.scatter(
        [str(n_tasks) for n_tasks in avg_n_tasks_shared_model["Number of nodes"]],
        avg_n_tasks_shared_model["Memory used"],
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax4.set_yscale("log", base=2)
    # ax4.set_xlabel("Number of nodes", fontweight="bold")
    ax4.set_ylim(
        [
            0.6 * results_df_no_shared_model["Memory used"].min(),
            1.6 * results_df_shared_model["Memory used"].max(),
        ]
    )
    ax4.grid(True)

    # Energy consumed used
    # No shared global model
    ax5.scatter(
        [str(n_tasks) for n_tasks in results_df_no_shared_model["Number of nodes"]],
        results_df_no_shared_model["Energy consumed"],
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Average
    ax5.scatter(
        [str(n_tasks) for n_tasks in avg_n_tasks_no_shared_model["Number of nodes"]],
        avg_n_tasks_no_shared_model["Energy consumed"],
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax5.set_ylim(
        [
            0.6 * results_df_no_shared_model["Energy consumed"].min(),
            1.6 * results_df_shared_model["Energy consumed"].max(),
        ]
    )
    ax5.set_yscale("log", base=2)
    ax5.set_ylabel("Energy / Wh", fontweight="bold")
    ax5.set_xlabel("Number of nodes", fontweight="bold")
    ax5.grid(True)
    energy_str = f"Overall {(energy_no_shared_model/1000):.2f} kWh consumed"
    ax5.text(
        0.05,
        0.95,
        energy_str,
        transform=ax5.transAxes,
        fontsize="x-small",
        verticalalignment="top",
        fontweight="bold",
    )

    # No shared global model
    ax6.scatter(
        [str(n_tasks) for n_tasks in results_df_shared_model["Number of nodes"]],
        results_df_shared_model["Energy consumed"],
        marker=".",
        color="k",
        zorder=10,
        alpha=0.5,
    )
    # Average
    ax6.scatter(
        [str(n_tasks) for n_tasks in avg_n_tasks_shared_model["Number of nodes"]],
        avg_n_tasks_shared_model["Energy consumed"],
        s=80,
        marker="X",
        facecolor="none",
        edgecolor="k",
        linewidths=1.3,
        zorder=20,
    )
    ax6.set_yscale("log", base=2)
    ax6.set_xlabel("Number of nodes", fontweight="bold")
    ax6.set_ylim(
        [
            0.6 * results_df_no_shared_model["Energy consumed"].min(),
            1.6 * results_df_shared_model["Energy consumed"].max(),
        ]
    )
    ax6.grid(True)
    energy_str = f"Overall {(energy_shared_model/1000):.2f} kWh consumed"
    ax6.text(
        0.05,
        0.95,
        energy_str,
        transform=ax6.transAxes,
        fontsize="x-small",
        verticalalignment="top",
        fontweight="bold",
    )
    # Ensure the layout is tight so labels don't overlap
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    #
    # Save the figure
    plt.savefig(
        pathlib.Path(root_dir_no_shared_model) / f"{data_set}_inference_flavor.png"
    )

    print(
        f"Overall energy consumed (no shared model): {(energy_no_shared_model/1000):.2f} kWh"
    )
    print(
        f"Overall energy consumed (shared model): {(energy_shared_model/1000):.2f} kWh"
    )

    # Show the plot
    plt.show()
