import argparse
import logging
import os
import pathlib
import re

import pandas as pd

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


def read_dataframe(path):
    log.debug(f'Parsing result csv: {path}')
    dataframe = pd.read_csv(path)

    if "accuracy_global_test" not in dataframe.columns:  # Serial run, slightly different data layout
        # global and local accuracies are the same, rename columns to match parallel results
        dataframe["accuracy_global_test"] = dataframe["accuracy_test"]
        dataframe["accuracy_local_test"] = dataframe["accuracy_test"]
        del dataframe["accuracy_test"]
        dataframe["accuracy_global_train"] = dataframe["accuracy_train"]
        dataframe["accuracy_local_train"] = dataframe["accuracy_train"]
        del dataframe["accuracy_train"]

    return dataframe


def extract_info_from_path(path, updated_path_names=False):
    log.debug(f'Extracting info from {path}')
    # Extract relevant information from the path.
    parts = str(path).split(os.sep)
    log.debug(parts)
    if updated_path_names:
        print(f"{parts[-2]=}, {parts[-1]=}")
        number_of_tasks = int(parts[-2].split("_")[2])
        model_seed = int(parts[-1].split("_")[1])  # Extract model seed from path.
    else:
        number_of_tasks = int(parts[-2].split("_")[1])
        model_seed = int(parts[-1].split("_")[2])  # Extract model seed from path.
    return number_of_tasks, model_seed


def extract_accuracies_from_csv(path):
    # Read the CSV file into a pandas dataframe.
    df = pd.read_csv(path)

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
    return local_test_accuracy_mean, local_test_accuracy_std, global_test_accuracy


def parse_log_file(path):
    log.debug(f'Parsing log file: {path}')
    with open(path, "r") as file:  # Load input text from the file.
        input_text = file.read()

    # Extract wall-clock time.
    pattern_wall_clock_time = r"Job Wall-clock time: (\d+:\d+:\d+|\d+-\d+:\d+:\d+)"
    time_match = re.search(pattern_wall_clock_time, input_text).group(1)
    wall_clock_time = time_to_seconds(time_match)  # type:ignore

    # Extract energy consumed.
    pattern_energy = r"(?<=\/ )\d+(\.\d+)?(?= Watthours)"
    energy_match = re.search(pattern_energy, input_text)
    energy_consumed = float(energy_match.group(0))  # type:ignore
    pattern_node_power_draw = r"Average node power draw: (\d+(\.\d+)?) Watt"
    avg_node_power_draw = float(re.search(pattern_node_power_draw, input_text).group(1))

    pattern_memory = r"Memory Utilized:\s*([0-9]+\.?[0-9]*)\s*(MB|GB|TB)"
    memory_match = re.search(pattern_memory, input_text)
    memory_utilized = memory_match.group(1)  # type:ignore
    unit = memory_match.group(2)  # type:ignore
    memory_in_gb = convert_to_gb(memory_utilized, unit)

    return {'wall_clock_time_sec': wall_clock_time, 'energy_consumed_watthours': energy_consumed,
            'memory_gb': memory_in_gb, 'avg_node_power_draw_watt': avg_node_power_draw}


def process_run_dir(path, dataset_label, updated_path_names=False):
    log.debug(f'Parsing dir {path}')
    number_of_tasks, model_seed = extract_info_from_path(path, updated_path_names=updated_path_names)
    csv_files = list(path.glob('*_results.csv'))
    log_files = list(path.glob('slurm*.out'))
    if not (len(csv_files) == 1 and len(log_files) == 1):
        log.warning(f'Found {len(csv_files)} csv files and {len(log_files)} log files '
                    f'in {path} instead of the expected one per type. Skipping directory.')
        return

    dataframe = read_dataframe(csv_files[0])
    parsed_from_log_file = parse_log_file(log_files[0])
    parsed_from_log_file['exp_node_power_draw'] = parsed_from_log_file['energy_consumed_watthours'] / (parsed_from_log_file['wall_clock_time_sec'] / 3600) / number_of_tasks
    run_key = ' '.join([path.parents[1].name, f"{number_of_tasks:2d}", str(model_seed), log_files[0].stem])
    parsed_values = ', '.join([f'{key}: {value:>10.2f}' for key, value in parsed_from_log_file.items()])
    log.info(f"Run {run_key}: {parsed_values}")

    dataframe["n_nodes"] = number_of_tasks
    dataframe["model_seed"] = model_seed

    for key, value in parsed_from_log_file.items():
        dataframe.loc[dataframe.comm_rank == "global", key] = value
    dataframe["dataset"] = dataset_label

    return dataframe


def process_experiment_dir(root_dir, scaling_type='strong', updated_path_names=False):
    root_dir = pathlib.Path(root_dir)
    dataset_label = root_dir.name

    # Dictionary to store the values grouped by (dataset, number_of_tasks, data seed)
    dataframes = []

    # Walk through the directory structure to find CSV files and extract accuracies
    for log_file in root_dir.glob("**/slurm*.out"):
        run_dir = log_file.parent
        log.debug(f"Currently parsing: {run_dir}")
        dataframes.append(process_run_dir(run_dir, dataset_label, updated_path_names=updated_path_names))
    results_df = pd.concat(dataframes)

    # Add mean local accuracies to global rank
    local_accuracies = [column for column in results_df.columns if re.match(r'accuracy_.*local.*', column)]
    aggregations = {column: "mean" for column in local_accuracies}
    key_columns = {'dataset', 'model_seed', 'n_nodes', 'mu_partition', 'mu_data'}
    key_columns = list(key_columns.intersection(set(results_df.columns)))
    mean_local_accuracies = results_df.groupby(key_columns, dropna=False).agg(aggregations).reset_index()
    mean_local_accuracies.rename(columns=lambda column: column.replace('local', 'mean_local'), inplace=True)
    results_df = pd.merge(results_df, mean_local_accuracies, on=key_columns)

    # Add serial runtimes and speedup/efficiency
    if scaling_type is not None:
        results_df = add_speedup_efficiency(results_df, scaling_type)

    log.info(f"List of columns in unaggregated dataframe: {list(results_df.columns)}")

    results_df.comm_size = results_df.comm_size.astype(int)

    results_df = results_df.sort_values(by=["dataset", "comm_size", "model_seed", "comm_rank"])

    return results_df


def add_speedup_efficiency(results_df, scaling_type):
    # Add serial runtimes (by model seed)
    time_columns = [column for column in results_df.columns if "time" in column]
    key_columns = ['dataset']
    if len(results_df[results_df.n_nodes == 1].model_seed.unique()) > 1:
        key_columns += ['model_seed']

    serial_times = results_df[results_df.n_nodes == 1][key_columns + time_columns]
    results_df = pd.merge(results_df, serial_times, on=key_columns, suffixes=('', '_serial'))

    # Add speedup/efficiency (by model seed)
    for parallel_column in time_columns:
        serial_column = parallel_column + '_serial'
        label = 'efficiency' if scaling_type == 'weak' else 'speedup'
        new_col_name = parallel_column.replace('time_sec', label).replace('time', label)
        log.debug(f'Adding column {new_col_name}')
        assert new_col_name not in results_df.columns
        results_df[new_col_name] = results_df[serial_column] / results_df[parallel_column]
    return results_df


def aggregate_by_seeds(dataframe, compute_std_for=None):
    # key columns used as keys for aggregation
    key_columns = ['comm_rank', 'n_samples', 'n_features', 'n_classes', 'train_split', 'n_trees', 'comm_size',
                   'n_nodes', 'dataset', 'shared_global_model']
    log.info('The following key columns are not in the dataframe, removing them from key columns: '
             f'{list(set(key_columns) - set(dataframe.columns))}')
    key_columns = [col for col in key_columns if col in dataframe.columns]
    # seed/run specific columns, ignored for aggregated dataframe
    seed_specific_columns = ['job_id', 'random_state', 'random_state_model', 'output_label', 'experiment_id',
                             'result_filename', 'model_seed', 'checkpoint_path', 'checkpoint_uid',
                             'n_clusters_per_class', 'frac_informative', 'frac_redundant',
                             'save_model', 'detailed_evaluation']

    # all remaining columns are aggregated
    value_columns = [column for column in dataframe.columns if column not in key_columns + seed_specific_columns]
    log.debug(f'Value columns for aggregation are: {value_columns}')

    compute_std_for = value_columns if compute_std_for == "all" else compute_std_for
    compute_std_for = compute_std_for or []

    mean_aggregations = {f"{column}_mean": (column, "mean") for column in value_columns}
    std_aggregations = {f"{column}_std": (column, "std") for column in compute_std_for}
    aggregations = {**mean_aggregations, **std_aggregations}

    return dataframe.groupby(key_columns, dropna=False).agg(**aggregations).reset_index()


if __name__ == "__main__":
    set_logger_config(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--scaling_type', type=str, choices=['strong', 'weak'])
    parser.add_argument('--updated_path_names', action='store_true')
    args = parser.parse_args()

    root_dir = pathlib.Path(args.data_dir)
    results_df = process_experiment_dir(root_dir, scaling_type=args.scaling_type, updated_path_names=args.updated_path_names)

    results_df.to_csv(root_dir / 'results.csv', index=False)
    log.info(f'Results written to {(root_dir / "results.csv").absolute()}')

    aggregated_results = aggregate_by_seeds(results_df, 'all')
    aggregated_results = aggregated_results.sort_values(by=["dataset", "comm_size", "comm_rank"])
    print_columns = ['dataset', 'comm_size', 'comm_rank',
                     'accuracy_global_test_mean', 'accuracy_mean_local_test_mean',
                     'wall_clock_time_sec_mean', 'time_sec_training_mean']
    if args.scaling_type == 'strong':
        print_columns += ['wall_clock_speedup_mean', 'speedup_training_mean']
    elif args.scaling_type == 'weak':
        print_columns += ['wall_clock_efficiency_mean', 'efficiency_training_mean']
    global_results = aggregated_results[aggregated_results.comm_rank == 'global']
    print(global_results[[col for col in print_columns if col in global_results.columns]])
    key_columns = ['comm_rank', 'n_samples', 'n_features', 'n_classes', 'n_trees', 'comm_size', 'n_nodes', 'dataset']
    print(global_results[key_columns])
    aggregated_results.to_csv(root_dir / 'aggregated_results.csv', index=False)
    log.info(f'Aggregated results written to {(root_dir / "aggregated_results.csv").absolute()}')

    with pd.option_context('display.float_format', '{:0.2f}'.format):
        print_columns = ['comm_size', 'model_seed', 'output_label', 'time_sec_training', 'wall_clock_time_sec', 'energy_consumed_watthours', 'avg_node_power_draw_watt', 'exp_node_power_draw']
        print(results_df[results_df.comm_rank == 'global'][[col for col in print_columns if col in results_df.columns]])

    overall_energy = results_df["energy_consumed_watthours"].sum()
    print(f'Overall energy consumed: {overall_energy.item()} watthours')

