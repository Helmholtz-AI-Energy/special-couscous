import os
import pathlib
import re
import sys
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

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

    # Walk through the directory structure to find CSV files
    for filename in pathlib.Path(root_dir).glob("**/*.csv"):
        # Extract relevant information from the directory structure
        print(f"Currently considered: {filename}")
        parts = str(filename).split(os.sep)
        number_of_tasks = int(
            parts[-3].split("_")[1]
        )  # Extract number of tasks from path.
        data_seed = int(parts[-2].split("_")[1])  # Extract data seed from path.
        model_seed = int(parts[-2].split("_")[2])  # Extract model seed from path.
        print(
            f"{data_set}: {number_of_tasks} tasks, data seed {data_seed}, model seed {model_seed}"
        )
        # Read the CSV file into a pandas dataframe.
        df = pd.read_csv(filename)

        # Extract the value from the target column and store it
        if "accuracy_global_test" in df.columns:  # Parallel runs
            global_test_accuracy = df.loc[
                df["comm_rank"] == "global", "accuracy_global_test"
            ].values[0]
            print(f"Global test accuracy: {global_test_accuracy}")
            results[(data_set, number_of_tasks, data_seed, model_seed)].append(
                global_test_accuracy
            )
        if "accuracy_test" in df.columns:  # Serial runs
            global_test_accuracy = df["accuracy_test"].values[0]
            print(f"Global test accuracy: {global_test_accuracy}")
            results[(data_set, number_of_tasks, data_seed, model_seed)].append(
                global_test_accuracy
            )

    for filename in pathlib.Path(root_dir).glob("**/*.out"):
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])
        data_seed = int(parts[-2].split("_")[1])  # Extract data seed from path.
        model_seed = int(parts[-2].split("_")[2])  # Extract model seed from path.
        with open(filename, "r") as file:  # Load input text from the file.
            input_text = file.read()
            print(f"Currently considered: {filename}")
        pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
        energy_match = re.search(pattern_energy, input_text)
        energy_consumed = float(energy_match.group(0))  # type:ignore
        print(f"Energy Consumed: {energy_consumed:.2f} Watthours")
        results[(data_set, number_of_tasks, data_seed, model_seed)].append(
            energy_consumed
        )

    # Save the results to a dataframe.
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
    # Return results sorted by number of tasks, data seed, and model seed.
    return results_df.sort_values(by=["Number of tasks", "Data seed", "Model seed"])


if __name__ == "__main__":
    # Get the root directory where results are stored from command line.
    root_dir = pathlib.Path(sys.argv[1])
    data_set = str(root_dir).split(os.sep)[-1]
    flavor = str(root_dir).split(os.sep)[-2]
    results_df = get_results_df(root_dir)
    # Calculate the overall energy consumed by this experiment series.
    overall_energy = results_df["Energy consumed"].sum()
    # Average results over model seeds for each data seed.
    avg_data_seeds = (
        results_df.groupby(["Number of tasks", "Data seed"])
        .agg({"Global test accuracy": "mean"})
        .reset_index()
    )
    print("Average results over model seeds for each data seed:\n", avg_data_seeds)
    # Average results over all seeds for each parallelization level.
    avg_n_tasks = (
        results_df.groupby(["Number of tasks"])
        .agg({"Global test accuracy": "mean"})
        .reset_index()
    )
    print(
        "Average results over all seeds for each parallelization level:\n", avg_n_tasks
    )

    f, ax = plt.subplots(figsize=(10, 6))
    # Settings
    visible = False
    plt.grid(visible)

    # Individual test acc. as small dots
    plt.scatter(
        [str(n_tasks) for n_tasks in results_df["Number of tasks"]],
        results_df["Global test accuracy"] * 100,
        label="Individual",
        marker=".",
        c=results_df["Data seed"],
        cmap="winter",
        zorder=10,
        alpha=0.5,
    )

    # Average test acc. over model seeds for each data seed as larger points
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

    # Overall average test acc. vs. number as stars
    plt.scatter(
        [str(n_tasks) for n_tasks in avg_n_tasks["Number of tasks"]],
        avg_n_tasks["Global test accuracy"] * 100,
        label="Average over all seeds for each number of tasks",
        s=200,
        marker="*",
        facecolor="none",
        edgecolor="firebrick",
        linewidths=1.3,
        zorder=20,
    )

    # Add labels and legend.
    plt.xlabel("Number of tasks", fontweight="bold")
    plt.ylabel("Test accuracy / %", fontweight="bold")
    data_set = data_set.replace("_", "")
    plt.title(
        f"{flavor.replace('_', ' ').capitalize()} {data_set}: Global test accuracy vs. number of tasks",
        fontweight="bold",
    )
    plt.legend(bbox_to_anchor=(0.325, -0.1), loc="upper left", fontsize="small")
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
    plt.tight_layout()
    plt.savefig(pathlib.Path(root_dir) / f"{data_set}_{flavor}_acc_drop.pdf")

    # Show the plot.
    plt.show()
