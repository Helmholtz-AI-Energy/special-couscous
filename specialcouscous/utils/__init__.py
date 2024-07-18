import datetime
import logging
import os
import pathlib
import re
import string
import sys
import time
import uuid
from typing import Any, Optional, Tuple, Union

import colorlog
import numpy as np
import pandas
from mpi4py import MPI

log = logging.getLogger(__name__)  # Get logger instance.


def get_problem_size(n: int, m: int, t: int) -> float:
    """
    Determine problem size from number of samples, number of features, and number of trees.

    Parameters
    ----------
    n : int
        Number of samples.
    m : int
        Number of features.
    t : int
        Number of trees.

    Returns
    -------
    float
       The problem size in terms of the random forest's time complexity.
    """
    return n * np.log2(n) * np.sqrt(m) * t


def set_logger_config(
    level: int = logging.INFO,
    log_file: Optional[Union[str, pathlib.Path]] = None,
    log_to_stdout: bool = True,
    log_rank: bool = False,
    colors: bool = True,
) -> None:
    """
    Set up the logger. Should only need to be done once. Generally, logging should only be done on the master rank.

    Parameters
    ----------
    level : int
        The default level for logging. Default is ``logging.INFO``.
    log_file : str | Path, optional
        The file to save the log to.
    log_to_stdout : bool
        A flag indicating if the log should be printed on stdout. Default is True.
    log_rank : bool
        A flag for prepending the MPI rank to the logging message. Default is False.
    colors : bool
        A flag for using colored logs. Default is True.
    """
    rank = f"{MPI.COMM_WORLD.Get_rank()}:" if log_rank else ""
    # Get base logger for SpecialCouscous.
    base_logger = logging.getLogger("specialcouscous")
    simple_formatter = logging.Formatter(
        f"{rank}:[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    if colors:
        formatter = colorlog.ColoredFormatter(
            fmt=f"{rank}[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
            f"[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
        )
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(formatter)
    else:
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(simple_formatter)

    if log_to_stdout:
        base_logger.addHandler(std_handler)
    if log_file is not None:
        log_file = pathlib.Path(log_file)
        log_dir = log_file.parents[0]
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(simple_formatter)
        base_logger.addHandler(file_handler)
    base_logger.setLevel(level)


class MPITimer:
    """
    MPI timer for measuring execution times in distributed environments over all ranks.

    Attributes
    ----------
    comm : MPI.Comm
        The MPI communicator to use.
    elapsed_time_average : float
        The average elapsed time in seconds.
    end_time : float
        The rank-local end time in seconds.
    name : str
        Label describing what this timer measured, can be used for printing the results.
    output_format : str
        Format string template used for printing the output. May reference all attributes of the timer.
    print_on_exit : bool
        Whether to print the measured time in ``__exit__``.
    start_time : float
        The rank-local start time in seconds.

    Methods
    -------
    start()
        Start the timer.
    stop()
        Stop the timer.
    allreduce_for_average()
        Compute the global average using allreduce.
    print()
        Print the elapsed time using the given template.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        print_on_exit: bool = True,
        name: str = "",
        output_format: str = "Elapsed time {name}: global average {elapsed_time_average:.2g}s, "
        "local {elapsed_time_local:.2g}s",
    ) -> None:
        """
        Create a new distributed context-manager enabled timer.

        Parameters
        ----------
        comm : MPI.Comm
            The MPI communicator.
        print_on_exit : bool
            Whether to print the measured time in ``__exit__``.
        name : str
            Label describing what this timer measured, can be used for printing the results.
        output_format : str
            Format string template used for printing the output. May reference all attributes of the timer.
        """
        self.comm = comm
        self.output_format = output_format
        self.print_on_exit = print_on_exit
        self.name = name

        # NOTE: In the constructor, the following variables are only partially initialized in terms of their types as
        # initializing their values with None causes problems with mypy.
        self.start_time: float
        self.end_time: float
        self.elapsed_time_local: float
        self.elapsed_time_average: float

    def start(self) -> None:
        """Start the timer by setting the start time."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer by setting the end time and updating ``elapsed_time_local``."""
        self.end_time = time.perf_counter()
        self.elapsed_time_local = self.end_time - self.start_time

    def allreduce_for_average_time(self) -> None:
        """Compute the global average using allreduce and update ``elapsed_time_average``."""
        self.elapsed_time_average = (
            self.comm.allreduce(self.elapsed_time_local, op=MPI.SUM) / self.comm.size
        )

    def print(self) -> None:
        """Print the elapsed time using the given template."""
        template_keywords = {
            key for (_, key, _, _) in string.Formatter().parse(self.output_format)
        }
        template_kwargs = {
            key: value for key, value in vars(self).items() if key in template_keywords
        }
        log.info(self.output_format.format(**template_kwargs))

    def __enter__(self) -> "MPITimer":
        """
        Start the timer.

        Called on entering the respective context (i.e., with a 'with' statement).

        Returns
        -------
        MPITimer
            This timer object.
        """
        self.start()
        return self

    def __exit__(self, *args: Tuple[Any, ...]) -> None:
        """
        Stop the timer, compute the global average, and optionally print the result on rank 0.

        Called on exiting the respective context (i.e., after a 'with' statement).

        Parameters
        ----------
        args : Any
            Unused, only to fulfill ``__exit__`` interface.
        """
        self.stop()
        self.allreduce_for_average_time()
        if self.print_on_exit and self.comm.rank == 0:
            self.print()


def construct_output_path(
    output_path: Union[pathlib.Path, str] = ".",
    output_name: str = "",
    experiment_id: str = "",
    mkdir: bool = True,
) -> Tuple[pathlib.Path, str]:
    """
    Construct the path and filename to save results to based on the current time and date.

    Returns an output directory: 'output_path / year / year-month / date / YYYY-mm-dd' or the subdirectory
    'output_path / year / year-month / date / YYYY-mm-dd / experiment_id' if an ``experiment_id`` is given and a
    base name for output files: 'YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>'.
    All directories on this path are created automatically unless ``mkdir`` is set to False.

    Parameters
    ----------
    output_path : Union[pathlib.Path, str]
        The path to the base output directory to create the date-based output directory tree in.
    output_name : str
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str
        If this is given, the file is placed in a further subdirectory of that name, i.e.,
        'output_path / year / year-month / date / experiment_id / <filename>.csv'.
        This can be used to group multiple runs of an experiment. Default is an empty string.
    mkdir : bool
        Whether to create all directories on the output path. Default is True.

    Returns
    -------
    Tuple[pathlib.Path, str]
        The path to the output directory and the base file name.
    """
    today = datetime.datetime.today()
    path = (
        pathlib.Path(output_path)
        / str(today.year)
        / f"{today.year}-{today.month}"
        / str(today.date())
    )
    if experiment_id != "":
        path /= experiment_id
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    base_filename = (
        f"{today.strftime('%Y-%m-%d--%H-%M-%S')}-{output_name}-{str(uuid.uuid4())[:8]}"
    )
    return path, base_filename


def save_dataframe(
    dataframe: pandas.DataFrame, output_path: Union[pathlib.Path, str]
) -> None:
    """
    Safe the given dataframe as csv to ``output_path``.

    If the path does not end with the '.csv' suffix, the suffix is appended to the filename.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe to save as csv.
    output_path : Union[pathlib.Path, str]
        The path to save to dataframe to.
    """
    output_path = pathlib.Path(output_path)
    if output_path.suffix != ".csv":
        output_path = output_path.parent / (output_path.name + ".csv")
    log.info(f"Saving results to {output_path.absolute()}")
    dataframe["result_filename"] = output_path
    dataframe.to_csv(output_path, index=False)


def dataframe_from_slurm_output(path: Union[pathlib.Path, str]) -> pandas.DataFrame:
    """
    Create a dataframe from SLURM output files and save it to csv file.

    Parameters
    ----------
    path : Union[pathlib.Path, str]
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


def time_to_seconds(time_str: str) -> Union[float, None]:
    """
    Convert wall-clock time string "d-hh:mm:ss" or "hh:mm:ss" into corresponding time in seconds.

    Parameters
    ----------
    time_str : str
        The wall-clock time string.

    Returns
    -------
    Union[float, None]
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
