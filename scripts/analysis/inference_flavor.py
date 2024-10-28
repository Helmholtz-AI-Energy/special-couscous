import os
import pathlib
import re
import sys
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_conf import ERROR_KWARGS, GLOBAL_ERROR_KWARGS, GLOBAL_LINE_KWARGS, LINE_KWARGS

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        The dataframe containing the average results for each parallelization level.
    pd.DataFrame
        The dataframe containing the results' standard deviation for each parallelization level.
    """
    # Dictionary to store the values grouped by (dataset, number_of_tasks, data seed)
    results = defaultdict(list)

    # Columns to extract from the CSV files:
    target_columns = [
        "time_sec_data_generation",
        "time_sec_forest_creation",
        "time_sec_training",
        "time_sec_all-gathering_model",
        "time_sec_test",
    ]

    # Loop over all CSV files in root directory.
    for filename in pathlib.Path(path_to_root).glob("**/*.csv"):
        print(f"Currently considered: {filename}")
        # Extract relevant information from the path.
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])  # Number of tasks
        model_seed = int(parts[-2].split("_")[2])  # Model seed
        # Read the CSV file into a dataframe.
        df = pd.read_csv(filename)

        # Extract the value from the target column and store it
        # Parallel runs:
        for column in target_columns:
            if column in df.columns:
                result = df.loc[df["comm_rank"] == "global", column].values[0]
            else:
                result = np.nan
            results[(data_set, number_of_tasks, model_seed)].append(result)

    # Loop over all SLURM output files in root directory.
    for filename in pathlib.Path(path_to_root).glob("**/*.out"):
        print(f"Currently considered: {filename}")
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])  # Number of tasks
        model_seed = int(parts[-2].split("_")[2])  # Model seed
        pattern_memory = r"Memory Utilized:\s*([0-9]+\.?[0-9]*)\s*(MB|GB|TB)"
        pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
        with open(filename, "r") as file:  # Load input text from the file.
            print(f"Currently considered: {filename}")
            input_text = file.read()
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
    std_n_tasks = (
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
        .std()
        .reset_index()
    )
    print(avg_n_tasks, std_n_tasks)
    return results_df, avg_n_tasks, std_n_tasks


if __name__ == "__main__":
    # Get the root directory where results are stored from command line.
    root_dir_no_shared_model = sys.argv[1]
    root_dir_shared_model = sys.argv[2]
    data_set = root_dir_no_shared_model.split(os.sep)[-1]

    results_df_no_shared_model, avg_no_shared_model, std_no_shared_model = (
        get_results_df(root_dir_no_shared_model)
    )
    energy_no_shared_model = results_df_no_shared_model["Energy consumed"].sum()
    results_df_shared_model, avg_shared_model, std_shared_model = get_results_df(
        root_dir_shared_model
    )
    energy_shared_model = results_df_shared_model["Energy consumed"].sum()
    # Create the figure and the first axis for test accuracy
    fig, axes = plt.subplots(3, 2, figsize=(5, 5), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    # Settings
    labelsize = "small"
    legendsize = "xx-small"
    visible = False  # Whether to plot a grid or not

    # Set title
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Inference flavor comparison {data_set}",
        fontweight="bold",
        fontsize="small",
    )
    ax1.set_title("No shared global model", fontsize="small")
    # Plot times over number of nodes for no shared global model.
    # Evaluate
    # Average
    ax1.errorbar(
        [str(n_tasks) for n_tasks in avg_no_shared_model["Number of nodes"]],
        avg_no_shared_model["Time for evaluation"] / 60,
        yerr=std_no_shared_model["Time for evaluation"] / 60,
        label="Evaluate model",
        **GLOBAL_ERROR_KWARGS,
    )
    ax1.plot(
        [str(n_tasks) for n_tasks in avg_no_shared_model["Number of nodes"]],
        avg_no_shared_model["Time for evaluation"] / 60,
        **GLOBAL_LINE_KWARGS,
    )
    ax1.set_ylim(
        [
            0.6 * avg_shared_model["Time for all-gathering model"].min() / 60,
            1.5 * avg_shared_model["Time for evaluation"].max() / 60,
        ]
    )
    ax1.set_yscale("log", base=2)
    ax1.set_ylabel("Time / min", fontweight="bold", fontsize=labelsize)
    ax1.grid(visible)
    # ax1.legend(loc="lower right", fontsize=legendsize)
    ax1.tick_params(axis="both", labelsize=labelsize)

    # Plot times over number of nodes for shared global model.
    ax2.set_title("Shared global model", fontsize="small")
    # Evaluate
    ax2.errorbar(
        [str(n_tasks) for n_tasks in avg_shared_model["Number of nodes"]],
        avg_shared_model["Time for evaluation"] / 60,
        yerr=std_shared_model["Time for evaluation"] / 60,
        label="Evaluate model",
        **GLOBAL_ERROR_KWARGS,
    )
    ax2.plot(
        [str(n_tasks) for n_tasks in avg_shared_model["Number of nodes"]],
        avg_shared_model["Time for evaluation"] / 60,
        **GLOBAL_LINE_KWARGS,
    )
    # All-gather global shared model.
    print(avg_shared_model["Time for all-gathering model"])
    ax2.errorbar(
        [str(n_tasks) for n_tasks in avg_shared_model["Number of nodes"]],
        avg_shared_model["Time for all-gathering model"] / 60,
        yerr=std_shared_model["Time for all-gathering model"] / 60,
        label="All-gather model",
        **ERROR_KWARGS,
    )
    ax2.plot(
        [str(n_tasks) for n_tasks in avg_shared_model["Number of nodes"]],
        avg_shared_model["Time for all-gathering model"] / 60,
        **LINE_KWARGS,
    )
    ax2.set_ylim(
        [
            0.6 * avg_shared_model["Time for all-gathering model"].min() / 60,
            1.5 * avg_shared_model["Time for evaluation"].max() / 60,
        ]
    )
    ax2.set_yscale("log", base=2)
    ax2.grid(visible)
    ax2.legend(loc="best", fontsize=legendsize)
    ax2.tick_params(axis="both", labelsize=labelsize)

    # Memory used
    # No shared global model
    ax3.errorbar(
        [str(n_tasks) for n_tasks in avg_no_shared_model["Number of nodes"]],
        avg_no_shared_model["Memory used"],
        yerr=std_no_shared_model["Memory used"],
        **GLOBAL_ERROR_KWARGS,
    )
    # Average
    ax3.plot(
        [str(n_tasks) for n_tasks in avg_no_shared_model["Number of nodes"]],
        avg_no_shared_model["Memory used"],
        **GLOBAL_LINE_KWARGS,
    )
    ax3.set_ylim(
        [
            0.6 * avg_no_shared_model["Memory used"].min(),
            1.6 * avg_no_shared_model["Memory used"].max(),
        ]
    )
    ax3.set_yscale("log", base=2)
    ax3.set_ylabel("Memory / GB", fontweight="bold", fontsize=labelsize)
    ax3.grid(visible)
    ax3.tick_params(axis="both", labelsize=labelsize)

    # Shared global model
    ax4.errorbar(
        [str(n_tasks) for n_tasks in avg_shared_model["Number of nodes"]],
        avg_shared_model["Memory used"],
        yerr=std_shared_model["Memory used"],
        **GLOBAL_ERROR_KWARGS,
    )
    ax4.plot(
        [str(n_tasks) for n_tasks in avg_shared_model["Number of nodes"]],
        avg_shared_model["Memory used"],
        **GLOBAL_LINE_KWARGS,
    )
    ax4.set_yscale("log", base=2)
    ax4.set_ylim(
        [
            0.6 * avg_no_shared_model["Memory used"].min(),
            1.6 * avg_shared_model["Memory used"].max(),
        ]
    )
    ax4.grid(visible)
    ax4.tick_params(axis="both", labelsize=labelsize)

    # Energy consumed used
    # No shared global model
    ax5.errorbar(
        [str(n_tasks) for n_tasks in avg_no_shared_model["Number of nodes"]],
        avg_no_shared_model["Energy consumed"],
        yerr=std_no_shared_model["Energy consumed"],
        **GLOBAL_ERROR_KWARGS,
    )
    ax5.plot(
        [str(n_tasks) for n_tasks in avg_no_shared_model["Number of nodes"]],
        avg_no_shared_model["Energy consumed"],
        **GLOBAL_LINE_KWARGS,
    )
    ax5.set_ylim(
        [
            0.6 * avg_no_shared_model["Energy consumed"].min(),
            1.6 * avg_shared_model["Energy consumed"].max(),
        ]
    )
    ax5.set_yscale("log", base=2)
    ax5.set_ylabel("Energy / Wh", fontweight="bold", fontsize=labelsize)
    ax5.set_xlabel("Number of nodes", fontweight="bold", fontsize=labelsize)
    ax5.grid(visible)
    energy_str = f"Overall {(energy_no_shared_model/1000):.2f} kWh consumed"
    ax5.text(
        0.05,
        0.95,
        energy_str,
        transform=ax5.transAxes,
        fontsize=legendsize,
        verticalalignment="top",
        fontweight="bold",
    )
    ax5.tick_params(axis="both", labelsize=labelsize)

    # No shared global model
    ax6.errorbar(
        [str(n_tasks) for n_tasks in avg_shared_model["Number of nodes"]],
        avg_shared_model["Energy consumed"],
        yerr=std_shared_model["Energy consumed"],
        **GLOBAL_ERROR_KWARGS,
    )
    ax6.plot(
        [str(n_tasks) for n_tasks in avg_shared_model["Number of nodes"]],
        avg_shared_model["Energy consumed"],
        **GLOBAL_LINE_KWARGS,
    )
    ax6.set_yscale("log", base=2)
    ax6.set_xlabel("Number of nodes", fontweight="bold", fontsize=labelsize)
    ax6.set_ylim(
        [
            0.6 * avg_no_shared_model["Energy consumed"].min(),
            1.6 * avg_shared_model["Energy consumed"].max(),
        ]
    )
    ax6.grid(visible)
    energy_str = f"Overall {(energy_shared_model/1000):.2f} kWh consumed"
    ax6.text(
        0.05,
        0.95,
        energy_str,
        transform=ax6.transAxes,
        fontsize=legendsize,
        verticalalignment="top",
        fontweight="bold",
    )
    ax6.tick_params(axis="both", labelsize=labelsize)

    plt.tight_layout()
    plt.savefig(
        pathlib.Path(root_dir_no_shared_model) / f"{data_set}_inference_flavor.pdf"
    )  # Save the figure.

    print(
        f"Overall energy consumed (no shared model): {(energy_no_shared_model/1000):.2f} kWh"
    )
    print(
        f"Overall energy consumed (shared model): {(energy_shared_model/1000):.2f} kWh"
    )
    plt.show()  # Show the plot.
