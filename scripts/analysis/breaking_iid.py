import os
import pathlib
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specialcouscous.utils.slurm import time_to_seconds

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


def get_results_df(root_dir: pathlib.Path | str) -> pd.DataFrame:
    """
    Construct breaking-IID results dataframe for plotting.

    Parameters
    ----------
    root_dir : Union[str, pathlib.Path]
        The path to the root directory containing the results.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the results.
    """
    # Dictionary to store the values grouped by (dataset, number_of_tasks, data seed)
    results = defaultdict(list)
    data_set = str(root_dir).split(os.sep)[-1]
    # Walk through the directory structure to find CSV files.
    for filename in pathlib.Path(root_dir).glob("**/*.csv"):
        print(f"Currently considered: {filename}")
        # Extract relevant information from the path.
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])
        data_seed = int(parts[-2].split("_")[1])
        model_seed = int(parts[-2].split("_")[2])  # Extract model seed from path.
        mu_data_str = str(parts[-2]).split("_")[3]
        if mu_data_str == "05":
            mu_data = 0.5
        elif mu_data_str == "inf":
            mu_data = np.inf
        elif mu_data_str == "20":
            mu_data = 2.0
        else:
            raise ValueError(f"Unknown mu_data: {mu_data_str}")
        mu_partition_str = parts[-2].split("_")[4]
        if mu_partition_str == "05":
            mu_partition = 0.5
        elif mu_partition_str == "inf":
            mu_partition = np.inf
        elif mu_partition_str == "20":
            mu_partition = 2.0
        else:
            raise ValueError(f"Unknown mu_partition: {mu_partition_str}")
        # Read the CSV file into a pandas dataframe.
        df = pd.read_csv(filename)

        # Extract the value from the target column and store it
        global_test_acc = df.loc[
            df["comm_rank"] == "global", "accuracy_global_test"
        ].values[0]
        local_test_acc_vals = tuple(df["accuracy_local_test"])
        local_test_acc_mean = df["accuracy_local_test"].dropna().mean()
        local_test_acc_std = df["accuracy_local_test"].dropna().std()
        time_train_vals = tuple(df["time_sec_training"])
        time_train_mean = df["time_sec_training"].dropna().mean()
        time_train_std = df["time_sec_training"].dropna().std()

        print(
            f"{data_set}: {number_of_tasks} tasks, data seed {data_seed}, model seed {model_seed}: "
            f"Global test acc.: {global_test_acc}"
        )
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(local_test_acc_vals)
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(local_test_acc_mean)
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(local_test_acc_std)
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(global_test_acc)
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(time_train_vals)
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(time_train_mean)
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(time_train_std)

    for filename in pathlib.Path(root_dir).glob("**/*.out"):
        print(f"Currently considered: {filename}")
        parts = str(filename).split(os.sep)
        number_of_tasks = int(parts[-3].split("_")[1])
        data_seed = int(parts[-2].split("_")[1])
        model_seed = int(parts[-2].split("_")[2])  # Extract model seed from path.
        mu_data_str = parts[-2].split("_")[3]
        if mu_data_str == "05":
            mu_data = 0.5
        elif mu_data_str == "inf":
            mu_data = np.inf
        elif mu_data_str == "20":
            mu_data = 2.0
        else:
            raise ValueError(f"Unknown mu_data: {mu_data_str}")
        mu_partition_str = parts[-2].split("_")[4]
        if mu_partition_str == "05":
            mu_partition = 0.5
        elif mu_partition_str == "inf":
            mu_partition = np.inf
        elif mu_partition_str == "20":
            mu_partition = 2.0
        else:
            raise ValueError(f"Unknown mu_partition: {mu_partition_str}")
        pattern_wall_clock_time = r"Job Wall-clock time: (\d+:\d+:\d+|\d+-\d+:\d+:\d+)"
        with open(filename, "r") as file:  # Load input text from the file.
            input_text = file.read()
        # Extract wall-clock time.
        wall_clock_time = time_to_seconds(
            re.search(pattern_wall_clock_time, input_text).group(1)  # type:ignore
        )
        print(f"Wall-clock time: {wall_clock_time} s")
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(wall_clock_time)  # type: ignore
        pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
        energy_match = re.search(pattern_energy, input_text)
        energy_consumed = float(energy_match.group(0))  # type:ignore
        print(f"Energy Consumed: {energy_consumed:.2f} Watthours")
        results[
            (data_set, number_of_tasks, data_seed, model_seed, mu_data, mu_partition)
        ].append(energy_consumed)  # type: ignore

    # Save the results to a dataframe.
    results_df = pd.DataFrame(
        [
            (
                k[0],
                k[2],
                k[3],
                k[4],
                k[5],
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6],
                v[7],
                v[8],
            )
            for k, v in results.items()
        ],
        columns=[
            "Dataset",
            "Data seed",
            "Model seed",
            "Mu data",
            "Mu partition",
            "Local test accuracy values",
            "Local test accuracy mean",
            "Local test accuracy error",
            "Global test accuracy",
            "Train time values",
            "Train time mean",
            "Train time error",
            "Wall-clock time",
            "Energy consumed",
        ],
    )
    return results_df.sort_values(by=["Model seed"]).reset_index(drop=True)


if __name__ == "__main__":
    # Get the root directory where results are stored from command line.
    root_dir = sys.argv[1]
    data_set = root_dir.split(os.sep)[-1]
    results_df = get_results_df(root_dir)
    print("Results dataframe:\n", results_df)

    # For each combination of data and partition imbalance, get average of test accuracy over model seeds.
    avg_results = results_df.groupby(["Mu data", "Mu partition"]).agg(
        {
            "Global test accuracy": "mean",
            "Local test accuracy mean": "mean",
            "Local test accuracy error": "mean",
            "Train time mean": "mean",
            "Train time error": "mean",
        }
    )

    print("Averaged results:\n", avg_results)
    overall_energy = results_df["Energy consumed"].sum()
    print(f"Overall energy consumed: {overall_energy} Wh")
    global_acc_std = results_df.groupby(["Mu data", "Mu partition"]).agg(
        {
            "Global test accuracy": "std",
        }
    )

    # Create the figure and the axes.
    nrows = 3
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(6, 5))
    # Settings
    global_errors = True
    labelsize = "x-small"
    legendsize = "xx-small"
    visible = False  # Whether to plot a grid or not
    mu_ticks = [0.5, 2.0, np.inf]
    mu_labels = ["0.5", "2.0", "inf"]

    # Set title.
    data_set = data_set.replace("_", "")
    plt.suptitle(
        f"Breaking IID {data_set}",
        fontweight="bold",
        fontsize="small",
    )

    # ----- AVERAGE GLOBAL TEST ACCURACY VS. MU_DATA AND MU_PARTITION
    global_acc_mean = np.array([avg_results["Global test accuracy"]]).reshape((3, 3))
    print("Global acc. mean:\n", global_acc_mean)
    im_global_mean = axs[0, 0].imshow(100 * global_acc_mean)

    # Show all ticks and label them with the respective list entries
    axs[0, 0].set_xlabel(r"$\mu_\text{partition}$", fontsize=labelsize)
    axs[0, 0].set_ylabel(r"$\mu_\text{data}$", fontsize=labelsize)
    axs[0, 0].set_xticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)
    axs[0, 0].set_yticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)

    # Create text annotations.
    for i in range(len(mu_ticks)):
        for j in range(len(mu_ticks)):
            axs[0, 0].text(
                j,
                i,
                f"{100 * global_acc_mean[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
                fontsize=labelsize,
            )
    cbar = fig.colorbar(im_global_mean, ax=axs[0, 0])
    cbar.set_label(label="Global accuracy\nmean / %", size=labelsize, weight="bold")
    cbar.ax.tick_params(labelsize=labelsize)

    # ----- AVERAGE GLOBAL TEST ACCURACY ERROR VS. MU_DATA AND MU_PARTITION
    global_acc_std = np.array([global_acc_std["Global test accuracy"]]).reshape((3, 3))
    im_global_std = axs[0, 1].imshow(100 * global_acc_std)

    # Show all ticks and label them with the respective list entries
    mu_ticks = [0.5, 2.0, np.inf]
    mu_labels = ["0.5", "2.0", "inf"]
    axs[0, 1].set_xlabel(r"$\mu_\text{partition}$", fontsize=labelsize)
    axs[0, 1].set_ylabel(r"$\mu_\text{data}$", fontsize=labelsize)
    axs[0, 1].set_xticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)
    axs[0, 1].set_yticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)

    # Loop over data dimensions and create text annotations.
    for i in range(len(mu_ticks)):
        for j in range(len(mu_ticks)):
            axs[0, 1].text(
                j,
                i,
                f"{100 * global_acc_std[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
                fontsize=labelsize,
            )
    cbar = fig.colorbar(im_global_std, ax=axs[0, 1])
    cbar.set_label(label="Global accuracy\nerror / %", size=labelsize, weight="bold")
    cbar.ax.tick_params(labelsize=labelsize)

    # ----- AVERAGE LOCAL TEST ACCURACY VS. MU_DATA AND MU_PARTITION
    local_acc_mean = np.array([avg_results["Local test accuracy mean"]]).reshape((3, 3))
    print("Local acc. mean:\n", local_acc_mean)
    im_local_mean = axs[1, 0].imshow(100 * local_acc_mean)

    # Show all ticks and label them with the respective list entries
    axs[1, 0].set_xlabel(r"$\mu_\text{partition}$", fontsize=labelsize)
    axs[1, 0].set_ylabel(r"$\mu_\text{data}$", fontsize=labelsize)
    axs[1, 0].set_xticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)
    axs[1, 0].set_yticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)

    # Loop over data dimensions and create text annotations.
    for i in range(len(mu_ticks)):
        for j in range(len(mu_ticks)):
            axs[1, 0].text(
                j,
                i,
                f"{100 * local_acc_mean[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
                fontsize=labelsize,
            )
    cbar = fig.colorbar(im_local_mean, ax=axs[1, 0])
    cbar.set_label(label="Local accuracy\nmean / %", size=labelsize, weight="bold")
    cbar.ax.tick_params(labelsize=labelsize)

    # ----- AVERAGE LOCAL TEST ACCURACY ERROR VS. MU_DATA AND MU_PARTITION
    if global_errors:
        local_acc_aggregated = results_df.groupby(["Mu data", "Mu partition"]).agg(
            {"Local test accuracy values": lambda x: sum(x, ())}
        )
        local_acc_std = np.array(
            local_acc_aggregated["Local test accuracy values"].apply(
                lambda x: pd.Series(x).std()
            )
        ).reshape((3, 3))
    else:
        local_acc_std = np.array([avg_results["Local test accuracy error"]]).reshape(
            (3, 3)
        )
    print("Local acc. std:\n", local_acc_std)
    im_local_std = axs[1, 1].imshow(100 * local_acc_std)

    # Show all ticks and label them with the respective list entries
    axs[1, 1].set_xlabel(r"$\mu_\text{partition}$", fontsize=labelsize)
    axs[1, 1].set_ylabel(r"$\mu_\text{data}$", fontsize=labelsize)
    axs[1, 1].set_xticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)
    axs[1, 1].set_yticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)

    # Loop over data dimensions and create text annotations.
    for i in range(len(mu_ticks)):
        for j in range(len(mu_ticks)):
            axs[1, 1].text(
                j,
                i,
                f"{100 * local_acc_std[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
                fontsize=labelsize,
            )
    cbar = fig.colorbar(im_local_std, ax=axs[1, 1])
    cbar.set_label(label="Local accuracy\nerror / %", size=labelsize, weight="bold")
    cbar.ax.tick_params(labelsize=labelsize)

    # ----- AVERAGE LOCAL TRAIN TIME VS. MU_DATA AND MU_PARTITION
    train_time_mean = np.array([avg_results["Train time mean"]]).reshape((3, 3)) / 60
    print("Local train times mean:\n", train_time_mean)
    im_train_mean = axs[2, 0].imshow(train_time_mean)

    # Show all ticks and label them with the respective list entries
    axs[2, 0].set_xlabel(r"$\mu_\text{partition}$", fontsize=labelsize)
    axs[2, 0].set_ylabel(r"$\mu_\text{data}$", fontsize=labelsize)
    axs[2, 0].set_xticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)
    axs[2, 0].set_yticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)

    # Create text annotations.
    for i in range(len(mu_ticks)):
        for j in range(len(mu_ticks)):
            text = axs[2, 0].text(
                j,
                i,
                f"{train_time_mean[i, j]:.1f}",
                ha="center",
                va="center",
                color="w",
                fontsize=labelsize,
            )
    cbar = fig.colorbar(im_train_mean, ax=axs[2, 0])
    cbar.set_label(label="Local train time\nmean / min", size=labelsize, weight="bold")
    cbar.ax.tick_params(labelsize=labelsize)

    # ----- AVERAGE LOCAL TRAIN TIME ERROR VS. MU_DATA AND MU_PARTITION
    if global_errors:
        train_times_aggregated = results_df.groupby(["Mu data", "Mu partition"]).agg(
            {"Train time values": lambda x: sum(x, ())}
        )
        train_time_std = np.array(
            train_times_aggregated["Train time values"].apply(
                lambda x: pd.Series(x).std()
            )
            / 60
        ).reshape((3, 3))
    else:
        train_time_std = (
            np.array([avg_results["Train time error"]]).reshape((3, 3)) / 60
        )
    print("Local train times std:\n", train_time_std)
    im_train_std = axs[2, 1].imshow(train_time_std)

    # Show all ticks and label them with the respective list entries
    axs[2, 1].set_xlabel(r"$\mu_\text{partition}$", fontsize=labelsize)
    axs[2, 1].set_ylabel(r"$\mu_\text{data}$", fontsize=labelsize)
    axs[2, 1].set_xticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)
    axs[2, 1].set_yticks(np.arange(len(mu_ticks)), labels=mu_labels, fontsize=labelsize)

    # Loop over data dimensions and create text annotations.
    for i in range(len(mu_ticks)):
        for j in range(len(mu_ticks)):
            text = axs[2, 1].text(
                j,
                i,
                f"{train_time_std[i, j]:.1f}",
                ha="center",
                va="center",
                color="w",
                fontsize=labelsize,
            )
    cbar = fig.colorbar(im_train_std, ax=axs[2, 1])
    cbar.set_label(label="Local train time\nerror / min", size=labelsize, weight="bold")
    cbar.ax.tick_params(labelsize=labelsize)

    plt.tight_layout()

    # Save the figure.
    if global_errors:
        figname = f"{data_set}_breaking_iid_global_errors.pdf"
    else:
        figname = f"{data_set}_breaking_iid.pdf"
    plt.savefig(pathlib.Path(root_dir) / figname)

    # Show the plot.
    plt.show()
