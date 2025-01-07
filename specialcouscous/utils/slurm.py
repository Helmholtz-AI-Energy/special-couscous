import os
import pathlib
import re

import pandas


def dataframe_from_slurm_output(path: pathlib.Path | str) -> pandas.DataFrame:
    """
    Create a dataframe from SLURM output files and save it to csv file.

    Parameters
    ----------
    path : pathlib.Path | str
        Path to folder with SLURM output files.

    Returns
    -------
    pandas.DataFrame
        The dataframe with (result) parameters of the provided SLURM jobs.
    """
    path = pathlib.Path(path)  # Convert to pathlib path.
    # Define regular expression patterns to extract each piece of information separately.
    pattern_samples = r"n_samples=(\d+)"
    pattern_features = r"n_features=(\d+)"
    pattern_trees = r"n_trees=(\d+)"
    pattern_training_time = r"Time for training is (\d+\.\d+) s\."
    pattern_accuracy = r"Accuracy is (\d+\.\d+)"
    pattern_wall_clock_time = r"Job Wall-clock time: (\d+:\d+:\d+|\d+-\d+:\d+:\d+)"
    pattern_energy_joule = r"Energy Consumed: (\d+) Joule"
    pattern_job_id = r"Job ID: (\d+)"
    pattern_job_state = r"State: (\w+)"

    # Initialize lists to store data from multiple files.
    n_samples_list = []
    n_features_list = []
    n_trees_list = []
    training_time_list = []
    accuracy_list = []
    wall_clock_time_list = []
    energy_joule_list = []
    job_id_list = []
    job_state_list = []

    for filename in os.listdir(
        path
    ):  # Loop through output files in the provided folder.
        if filename.endswith(".out"):
            file_path = os.path.join(path, filename)  # Construct full file path.
            with open(file_path, "r") as file:  # Load input text from the file.
                input_text = file.read()

            # Extract information using regular expressions.
            n_samples_match = re.search(pattern_samples, input_text)
            n_features_match = re.search(pattern_features, input_text)
            n_trees_match = re.search(pattern_trees, input_text)
            training_time_match = re.search(pattern_training_time, input_text)
            accuracy_match = re.search(pattern_accuracy, input_text)
            wall_clock_time_match = re.search(pattern_wall_clock_time, input_text)
            energy_joule_match = re.search(pattern_energy_joule, input_text)
            job_id_match = re.search(pattern_job_id, input_text)
            job_state_match = re.search(pattern_job_state, input_text)

            # Append extracted data to their respective lists ("N/A" if not found).
            n_samples_list.append(
                int(n_samples_match.group(1)) if n_samples_match else "N/A"
            )
            n_features_list.append(
                int(n_features_match.group(1)) if n_features_match else "N/A"
            )
            n_trees_list.append(int(n_trees_match.group(1)) if n_trees_match else "N/A")
            training_time_list.append(
                float(training_time_match.group(1)) if training_time_match else "N/A"
            )
            accuracy_list.append(
                float(accuracy_match.group(1)) if accuracy_match else "N/A"
            )
            wall_clock_time_list.append(
                wall_clock_time_match.group(1) if wall_clock_time_match else "N/A"
            )
            energy_joule_list.append(
                int(energy_joule_match.group(1)) if energy_joule_match else "N/A"
            )
            job_id_list.append(int(job_id_match.group(1)) if job_id_match else "N/A")
            job_state_list.append(
                job_state_match.group(1) if job_state_match else "N/A"
            )

    # Create a pandas dataframe from the extracted data.
    data = {
        "n_samples": n_samples_list,
        "n_features": n_features_list,
        "n_trees": n_trees_list,
        "wall_clock_time": wall_clock_time_list,
        "training_time": training_time_list,
        "accuracy": accuracy_list,
        "energy_joule": energy_joule_list,
        "job_id": job_id_list,
        "job_state": job_state_list,
    }
    df = pandas.DataFrame(data)
    df.sort_values(
        by=["n_trees", "n_samples", "n_features"],
        ascending=[False, True, True],
        inplace=True,
        ignore_index=True,
    )

    df["data_entries"] = df["n_samples"] * df["n_features"]
    df["n_samples"] = df["n_samples"].apply(lambda x: "{:.0E}".format(x))
    df["n_features"] = df["n_features"].apply(lambda x: "{:.0E}".format(x))
    df["n_trees"] = df["n_trees"].apply(lambda x: "{:.0E}".format(x))
    df["data_entries"] = df["data_entries"].apply(lambda x: "{:.0E}".format(x))

    df.to_csv(path / pathlib.Path("results.csv"))  # Save dataframe to csv file.
    return df


def time_to_seconds(time_str: str) -> float | None:
    """
    Convert wall-clock time string "d-hh:mm:ss" or "hh:mm:ss" into corresponding time in seconds.

    Parameters
    ----------
    time_str : str
        The wall-clock time string.

    Returns
    -------
    float | None
        The wall-clock time in seconds (None if provided string was invalid).
    """
    time_pattern = r"(\d+)-(\d+):(\d+):(\d+)|(\d+):(\d+):(\d+)"  # Define regular expression to match time strings.
    match = re.match(time_pattern, time_str)  # Match the time string using the pattern.

    if match:
        # Extract hours, minutes, and seconds from matched groups.
        if match.group(1):
            days, hours, minutes, seconds = map(int, match.group(1, 2, 3, 4))
        else:
            days = 0
            hours, minutes, seconds = map(int, match.group(5, 6, 7))

        total_seconds = (
            (days * 24 * 60 * 60) + (hours * 60 * 60) + (minutes * 60) + seconds
        )  # Calculate total time in seconds.
        return float(total_seconds)
    else:
        return None  # Return None for invalid time strings.


def expand_node_range(s: str) -> list[str]:
    """
    Expand a range string of compute nodes into a list of individual nodes.

    Parameters
    ----------
    s : str
        The string to expand into a list of compute nodes, as extracted from the job output file.

    Returns
    -------
    list[str]
        The expanded list of compute nodes.
    """
    nodes = []
    # Extract the content inside the square brackets
    match = re.search(r"\[(.*?)\]", s)
    if match:
        ranges = match.group(1).split(",")
        for r in ranges:
            if "-" in r:  # Handle ranges like 0041-0042
                start, end = r.split("-")
                nodes.extend([f"{int(i):04}" for i in range(int(start), int(end) + 1)])
            else:  # Handle individual nodes like 0016
                nodes.append(f"{int(r):04}")
    return nodes


def find_checkpoint_dir_and_uuid(
    base_path: pathlib.Path,
    log_n_samples: int,
    log_n_features: int,
    mu_global: float | str,
    mu_local: float | str,
    data_seed: int,
    model_seed: int,
) -> tuple[pathlib.Path, str]:
    """
    Find the checkpoint directory and extract the UUID from the filenames.

    Parameters
    ----------
    base_path : pathlib.Path
        The base path where results are stored.
    log_n_samples : int
        The common logarithm of the number of samples to use.
    log_n_features : int
        The common logarithm of the number of features to use.
    mu_global : float | str
        The global imbalance factor.
    mu_local : float | str
        The local imbalance factor.
    data_seed : int
        The random state used for synthetic dataset generation, splitting, and distribution.
    model_seed : int
        The (base) random state used for initializing the (distributed) model.

    Returns
    -------
    tuple[pathlib.Path, str]
        A tuple containing the path to the checkpoint directory and the UUID.
    """
    # Convert parameters to match the directory naming convention.
    mu_global_str = str(mu_global).replace(".", "")
    mu_local_str = str(mu_local).replace(".", "")

    # Construct the expected directory path pattern
    search_pattern = (
        f"breaking_iid/n{log_n_samples}_m{log_n_features}/nodes_16/"
        f"*_{data_seed}_{model_seed}_{mu_global_str}_{mu_local_str}/"
    )
    print(f"The search pattern is {search_pattern}.")

    # Search for matching directories.
    matching_dirs = list(base_path.glob(search_pattern))

    if len(matching_dirs) == 0:
        raise FileNotFoundError(
            f"No checkpoint directory found for the specified parameters in {base_path}."
        )
    elif len(matching_dirs) > 1:
        raise ValueError(f"Multiple checkpoint directories found: {matching_dirs}")

    checkpoint_dir = matching_dirs[0]

    # Extract the UUID from the results file.
    uuid_pattern = re.compile(r"--([\d\w-]+)_results\.csv$")
    for file in checkpoint_dir.iterdir():
        if file.name.endswith("_results.csv"):
            match = uuid_pattern.search(file.name)
            if match:
                return checkpoint_dir, match.group(1)

    raise ValueError(f"No UUID found in results files within {checkpoint_dir}")
