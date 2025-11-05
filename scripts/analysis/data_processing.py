import argparse
import logging
import os
import pathlib
import re
from typing import Any

import numpy as np
import pandas
import pandas as pd
import scipy

from specialcouscous.utils import set_logger_config
from specialcouscous.utils.slurm import time_to_seconds

pd.set_option("display.max_rows", None)

log = logging.getLogger("specialcouscous.data_processing")


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


def read_dataframe(path: str | os.PathLike) -> pandas.DataFrame:
    """
    Read result data from csv and add global/local accuracy/auc columns for serial runs.

    Parameters
    ----------
    path : str | os.PathLike
        The path to the result csv to read.

    Returns
    -------
    pandas.DataFrame
        The result data read from the csv with global/local columns.
    """
    log.debug(f"Parsing result csv: {path}")
    dataframe = pd.read_csv(path)

    if (
        "accuracy_global_test" not in dataframe.columns
    ):  # Serial run, slightly different data layout
        # global and local accuracies are the same, rename columns to match parallel results
        dataframe["accuracy_global_test"] = dataframe["accuracy_test"]
        dataframe["accuracy_local_test"] = dataframe["accuracy_test"]
        del dataframe["accuracy_test"]
        dataframe["accuracy_global_train"] = dataframe["accuracy_train"]
        dataframe["accuracy_local_train"] = dataframe["accuracy_train"]
        del dataframe["accuracy_train"]
        if "auc_test" in dataframe:
            dataframe["auc_global_test"] = dataframe["auc_test"]
            dataframe["auc_local_test"] = dataframe["auc_test"]
            del dataframe["auc_test"]
        if "auc_train" in dataframe:
            dataframe["auc_global_train"] = dataframe["auc_train"]
            dataframe["auc_local_train"] = dataframe["auc_train"]
            del dataframe["auc_train"]

    return dataframe


def extract_info_from_path(
    path: str | os.PathLike, updated_path_names: bool = False
) -> tuple[int, int]:
    """
    Parse the given path to a result directory, extracting the number of tasks (compute nodes) and the model seed.

    Original path structure:
    .../nodes_<number of tasks>/<job id>_<data seed>_<model seed>
    Updated path structure:
    .../n_nodes_<number of tasks>/<data seed>_<model seed>_<job id>

    Parameters
    ----------
    path : str | os.PathLike
        The path to parse.
    updated_path_names : bool
        Compatibility option to switch between the original and updated path names (see above).

    Returns
    -------
    int
        The number of tasks (= number of compute nodes).
    int
        The model seed.
    """
    log.debug(f"Extracting info from {path}")
    # Extract relevant information from the path.
    parts = str(path).split(os.sep)
    log.debug(parts)
    if updated_path_names:
        log.debug(f"{parts[-2]=}, {parts[-1]=}")
        number_of_tasks = int(parts[-2].split("_")[2])
        model_seed = int(parts[-1].split("_")[1])  # Extract model seed from path.
    else:
        number_of_tasks = int(parts[-2].split("_")[1])
        model_seed = int(parts[-1].split("_")[2])  # Extract model seed from path.
    return number_of_tasks, model_seed


def parse_log_file(path: str | os.PathLike) -> dict[str, float]:
    """
    Parse a slurm log file to extract wall-clock time and memory and energy consumption.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a slurm log file.

    Returns
    -------
    dict[str, float]
        A dict with the following string keys and float values:
        - wall_clock_time_sec: the wall-clock time in seconds
        - energy_consumed_watthours: the consumed energy in watthours
        - memory_gb: the maximum memory in GB
        - avg_node_power_draw_watt: the average node power draw in watt
    """
    log.debug(f"Parsing log file: {path}")
    with open(path, "r") as file:  # Load input text from the file.
        input_text = file.read()

    # Extract wall-clock time.
    pattern_wall_clock_time = r"Job Wall-clock time: (\d+:\d+:\d+|\d+-\d+:\d+:\d+)"
    time_match = re.search(pattern_wall_clock_time, input_text)
    wall_clock_time = time_to_seconds(time_match.group(1)) if time_match else np.nan

    # Extract energy consumed.
    pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
    energy_match = re.search(pattern_energy, input_text)
    energy_consumed = float(energy_match.group(0)) if energy_match else np.nan
    pattern_node_power_draw = r"Average node power draw: (\d+(\.\d+)?) Watt"
    power_draw_match = re.search(pattern_node_power_draw, input_text)
    avg_node_power_draw = (
        float(power_draw_match.group(1)) if power_draw_match else np.nan
    )

    pattern_memory = r"Memory Utilized:\s*([0-9]+\.?[0-9]*)\s*(MB|GB|TB)"
    memory_match = re.search(pattern_memory, input_text)
    if memory_match:
        memory_utilized = memory_match.group(1)  # type:ignore
        unit = memory_match.group(2)  # type:ignore
        memory_in_gb = convert_to_gb(memory_utilized, unit)
    else:
        memory_in_gb = np.nan

    return {
        "wall_clock_time_sec": wall_clock_time,
        "energy_consumed_watthours": energy_consumed,
        "memory_gb": memory_in_gb,
        "avg_node_power_draw_watt": avg_node_power_draw,
    }


def process_run_dir(
    path: pathlib.Path, dataset_label: Any, updated_path_names: bool = False
) -> pandas.DataFrame | None:
    """
    Process the results of one experiment run, combining the information from the path name, csv, and log file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the directory containing the csv and log file for this run.
    dataset_label : Any
        If the csv contains no column "dataset", a new column with this value is added.
    updated_path_names : bool
        Compatibility option to switch between original and updated path names, see extract_info_from_path.

    Returns
    -------
    pandas.DataFrame | None
        The parse dataframe or None if parsing failed.
    """
    log.debug(f"Parsing dir {path}")
    number_of_tasks, model_seed = extract_info_from_path(
        path, updated_path_names=updated_path_names
    )
    csv_files = list(path.glob("*_results.csv"))
    log_files = list(path.glob("slurm*.out"))
    if not (len(csv_files) == 1 and len(log_files) == 1):
        log.warning(
            f"Found {len(csv_files)} csv files and {len(log_files)} log files "
            f"in {path} instead of the expected one per type. Skipping directory."
        )
        return None

    dataframe = read_dataframe(csv_files[0])
    parsed_from_log_file = parse_log_file(log_files[0])
    parsed_from_log_file["exp_node_power_draw"] = (
        parsed_from_log_file["energy_consumed_watthours"]
        / (parsed_from_log_file["wall_clock_time_sec"] / 3600)
        / number_of_tasks
    )
    run_key = " ".join(
        [
            path.parents[1].name,
            f"{number_of_tasks:2d}",
            str(model_seed),
            log_files[0].stem,
        ]
    )
    parsed_values = ", ".join(
        [f"{key}: {value:>10.2f}" for key, value in parsed_from_log_file.items()]
    )
    log.info(f"Run {run_key}: {parsed_values}")

    dataframe["n_nodes"] = number_of_tasks
    dataframe["model_seed"] = model_seed

    for key, value in parsed_from_log_file.items():
        dataframe.loc[dataframe.comm_rank == "global", key] = value
    if "dataset" not in dataframe.columns:
        dataframe["dataset"] = dataset_label

    return dataframe


def process_experiment_dir(
    root_dir: str | os.PathLike,
    scaling_type: None | str = "strong",
    updated_path_names: bool = False,
) -> pandas.DataFrame:
    """
    Process a directory hierarchy for an experiment consisting of multiple runs.

    Try to parse all subdirectories of the root path containing a slurm log file (slurm*.out) with process_run_dir,
    skipping those than fail to parse (e.g. no or too many result csvs).
    Combine all resulting dataframes and aggregate rank-local accuracy/AUC scores.
    Optionally add efficiency or speedup depending on the given scaling_type.

    Parameters
    ----------
    root_dir : str | os.PathLike
        The root of the directory hierarchy to parse.
    scaling_type : None | str
        If not None, compute speedup or efficiency see add_speedup_efficiency. In that case, serial results
        (n_nodes == 1) are required.
    updated_path_names : bool
        Compatibility option to switch between original and updated path names, see extract_info_from_path.

    Returns
    -------
    pandas.DataFrame
        The dataframe containing the results for all runs in this experiment.
    """
    root_dir = pathlib.Path(root_dir)
    dataset_label = root_dir.name

    # Dictionary to store the values grouped by (dataset, number_of_tasks, data seed)
    dataframes = []

    # Walk through the directory structure to find CSV files and extract accuracies
    for log_file in root_dir.glob("**/slurm*.out"):
        run_dir = log_file.parent
        log.debug(f"Currently parsing: {run_dir}")
        dataframes.append(
            process_run_dir(
                run_dir, dataset_label, updated_path_names=updated_path_names
            )
        )
    results_df = pd.concat(dataframes)

    # Add mean local accuracies to global rank
    local_accuracies = [
        column
        for column in results_df.columns
        if re.match(r"(accuracy|auc)_.*local.*", column)
    ]
    aggregations = {column: "mean" for column in local_accuracies}
    key_columns = ["dataset", "model_seed", "n_nodes", "mu_partition", "mu_data"]
    key_columns = list(set(key_columns).intersection(set(results_df.columns)))
    mean_local_accuracies = (
        results_df.groupby(key_columns, dropna=False).agg(aggregations).reset_index()
    )
    mean_local_accuracies.rename(
        columns=lambda column: column.replace("local", "mean_local"), inplace=True
    )
    results_df = pd.merge(results_df, mean_local_accuracies, on=key_columns)
    results_df["trees_per_node"] = results_df.n_trees / results_df.n_nodes

    # Add serial runtimes and speedup/efficiency
    if scaling_type is not None:
        results_df = add_speedup_efficiency(results_df, scaling_type)

    log.info(f"List of columns in unaggregated dataframe: {list(results_df.columns)}")

    results_df.comm_size = results_df.comm_size.astype(int)

    results_df = results_df.sort_values(
        by=["dataset", "comm_size", "model_seed", "comm_rank"]
    )

    return results_df


def add_speedup_efficiency(
    results_df: pandas.DataFrame, scaling_type: str
) -> pandas.DataFrame:
    """
    Add speedup or efficiency columns for all time measurements.

    All columns in the given dataframe containing "time" are considered time measurements.
    First matches all parallel runs with the corresponding serial runs (by matching the dataset, model_seed (if
    multiple), and trees_per_node (weak scaling) / n_trees (strong scaling). Then computes efficiency/speedup by
    dividing the serial runtimes by their parallel counterparts.

    Parameters
    ----------
    results_df : pandas.DataFrame
        The results for all runs in the current experiment.
        Needs to contain the columns n_nodes, dataset, model_seed, n_trees, trees_per_node and matching serial run
        (n_nodes == 1) for all parallel runs.
    scaling_type : str
        Switch between weak scaling ("weak", compute efficiency) or strong scaling ("strong", compute speedup).

    Returns
    -------
    pandas.DataFrame
        The input dataframe with added speedup/efficiency columns for all time measurements.
    """
    # Add serial runtimes (by model seed)
    time_columns = [column for column in results_df.columns if "time" in column]
    key_columns = ["dataset", "trees_per_node" if scaling_type == "weak" else "n_trees"]
    if len(results_df[results_df.n_nodes == 1].model_seed.unique()) > 1:
        key_columns += ["model_seed"]

    serial_times = results_df[results_df.n_nodes == 1][key_columns + time_columns]
    results_df = pd.merge(
        results_df, serial_times, on=key_columns, suffixes=("", "_serial")
    )

    # Add speedup/efficiency (by model seed)
    for parallel_column in time_columns:
        serial_column = parallel_column + "_serial"
        label = "efficiency" if scaling_type == "weak" else "speedup"
        new_col_name = parallel_column.replace("time_sec", label).replace("time", label)
        log.debug(f"Adding column {new_col_name}")
        assert new_col_name not in results_df.columns
        results_df[new_col_name] = (
            results_df[serial_column] / results_df[parallel_column]
        )
    return results_df


def aggregate_by_seeds(
    dataframe: pandas.DataFrame, compute_std_for: None | str | list[str] = None
) -> pandas.DataFrame:
    """
    Aggregate the results over multiple seeds, computing mean, standard deviation, and 95% confidence interval.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Experimental results for potentially multiple model and data seeds.
    compute_std_for : None | str | list[str]
        The columns to compute standard deviation and 95% CI for. Can be either a list of column names, "all" to
        indicate all value columns, or None to indicate no columns.

    Returns
    -------
    pandas.DataFrame
        The input dataset aggregated over all seeds.
    """
    # key columns used as keys for aggregation
    key_columns = [
        "comm_rank",
        "n_samples",
        "n_features",
        "n_classes",
        "train_split",
        "n_trees",
        "comm_size",
        "n_nodes",
        "dataset",
        "shared_global_model",
    ]
    log.info(
        "The following key columns are not in the dataframe, removing them from key columns: "
        f"{list(set(key_columns) - set(dataframe.columns))}"
    )
    key_columns = [col for col in key_columns if col in dataframe.columns]
    # seed/run specific columns, ignored for aggregated dataframe
    seed_specific_columns = [
        "job_id",
        "random_state",
        "random_state_model",
        "output_label",
        "experiment_id",
        "result_filename",
        "model_seed",
        "checkpoint_path",
        "checkpoint_uid",
        "n_clusters_per_class",
        "frac_informative",
        "frac_redundant",
        "save_model",
        "detailed_evaluation",
        "data_dir",
        "stratified_train_test",
    ]

    # all remaining columns are aggregated
    value_columns = [
        column
        for column in dataframe.columns
        if column not in key_columns + seed_specific_columns
    ]
    log.debug(f"Value columns for aggregation are: {value_columns}")

    compute_std_for = value_columns if compute_std_for == "all" else compute_std_for
    compute_std_for = compute_std_for or []

    def mean_confidence_interval(
        data: pandas.Series, confidence: float = 0.95
    ) -> pandas.Series:
        return scipy.stats.sem(data) * scipy.stats.t.ppf(
            (1 + confidence) / 2.0, len(data) - 1
        )

    mean_aggregations = {f"{column}_mean": (column, "mean") for column in value_columns}
    std_aggregations = {f"{column}_std": (column, "std") for column in compute_std_for}
    ci_aggregations = {
        f"{column}_ci95": (column, mean_confidence_interval)
        for column in compute_std_for
    }
    aggregations = {**mean_aggregations, **std_aggregations, **ci_aggregations}

    return (
        dataframe.groupby(key_columns, dropna=False).agg(**aggregations).reset_index()
    )


if __name__ == "__main__":
    set_logger_config(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--scaling_type", type=str, choices=["strong", "weak"])
    parser.add_argument("--updated_path_names", action="store_true")
    args = parser.parse_args()

    root_dir = pathlib.Path(args.data_dir)
    results_df = process_experiment_dir(
        root_dir,
        scaling_type=args.scaling_type,
        updated_path_names=args.updated_path_names,
    )

    results_df.to_csv(root_dir / "results.csv", index=False)
    log.info(f"Results written to {(root_dir / 'results.csv').absolute()}")

    aggregated_results = aggregate_by_seeds(results_df, "all")
    aggregated_results = aggregated_results.sort_values(
        by=["dataset", "comm_size", "comm_rank"]
    )
    print_columns = [
        "dataset",
        "comm_size",
        "comm_rank",
        "accuracy_global_test_mean",
        "accuracy_mean_local_test_mean",
        "wall_clock_time_sec_mean",
        "time_sec_training_mean",
    ]
    if args.scaling_type == "strong":
        print_columns += ["wall_clock_speedup_mean", "speedup_training_mean"]
    elif args.scaling_type == "weak":
        print_columns += ["wall_clock_efficiency_mean", "efficiency_training_mean"]
    global_results = aggregated_results[aggregated_results.comm_rank == "global"]
    print(
        global_results[[col for col in print_columns if col in global_results.columns]]
    )
    key_columns = [
        "comm_rank",
        "n_samples",
        "n_features",
        "n_classes",
        "n_trees",
        "comm_size",
        "n_nodes",
        "dataset",
    ]
    key_columns = [col for col in key_columns if col in global_results.columns]
    print(global_results[key_columns])
    aggregated_results.to_csv(root_dir / "aggregated_results.csv", index=False)
    log.info(
        f"Aggregated results written to {(root_dir / 'aggregated_results.csv').absolute()}"
    )

    with pd.option_context("display.float_format", "{:0.2f}".format):
        print_columns = [
            "comm_size",
            "model_seed",
            "output_label",
            "time_sec_training",
            "wall_clock_time_sec",
            "energy_consumed_watthours",
            "avg_node_power_draw_watt",
            "exp_node_power_draw",
        ]
        print(
            results_df[results_df.comm_rank == "global"][
                [col for col in print_columns if col in results_df.columns]
            ]
        )

    overall_energy = results_df["energy_consumed_watthours"].sum()
    print(f"Overall energy consumed: {overall_energy.item()} watthours")
