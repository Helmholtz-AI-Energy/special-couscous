import datetime
import itertools
import os
import pathlib
import re
import string
import time
import uuid
from typing import Collection, List, Optional, Tuple, Dict, Union

import pandas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from mpi4py import MPI


class MPITimer:
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
            Whether to print the measured time in __exit__.
        name : str
            Label describing what this timer measured, can be used for printing the results.
        output_format : str
            Format string template used for printing the output. May reference all attributes of the
        """
        self.comm = comm
        self.output_format = output_format
        self.print_on_exit = print_on_exit
        self.name = name

        self.start_time = None
        self.end_time = None
        self.elapsed_time_local = None
        self.elapsed_time_average = None

    def start(self) -> None:
        """Start the timer by setting the start time."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer by setting the end time and updating elapsed_time_local."""
        self.end_time = time.perf_counter()
        self.elapsed_time_local = self.end_time - self.start_time

    def allreduce_for_average_time(self) -> None:
        """Compute the global average using allreduce and update elapsed_time_average."""
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
        print(self.output_format.format(**template_kwargs))

    def __enter__(self) -> "MPITimer":
        """
        Called on entering this context (e.g. with a 'with' statement), starts the timer.

        Returns
        -------
        MPITimer
            This timer object.
        """
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """
        Called on exiting this context (e.g. after a 'with' statement), stops the timer, computes the global average
        and optionally prints the result on rank 0.

        Parameters
        ----------
        args :
            Unused, only to fulfill __exit__ interface.
        """
        self.stop()
        self.allreduce_for_average_time()
        if self.print_on_exit and self.comm.rank == 0:
            self.print()


def construct_output_path(
    output_path: Union[pathlib.Path, str] = ".",
    output_name: str = "",
    experiment_id: str = None,
    mkdir: bool = True,
) -> Tuple[pathlib.Path, str]:
    """
    Constructs the path and filename to save results at based on the current time and date.
    Returns an output directory: 'output_path / year / year-month / date / YYYY-mm-dd' or the subdirectory
    'output_path / year / year-month / date / YYYY-mm-dd / experiment_id' if an experiment_id is given and a base_name
    for output files: 'YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>'.
    All directories on this path are created automatically unless mkdir is set to False.

    Parameters
    ----------
    output_path : Union[pathlib.Path, str]
        The path to the base output directory to create the date-based output directory tree in.
    output_name : str
        Optional label for the csv file, added to the name after the timestamp.
    experiment_id : str
        If this is given, the file is placed in a further subdirectory of that name, i.e.,
        output_path / year / year-month / date / experiment_id / <filename>.csv
        This can be used to group multiple runs of an experiment.
    mkdir : bool
        Whether to create all directories on the output path.

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
    if experiment_id:
        path /= experiment_id
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    base_filename = (
        f"{today.strftime('%Y-%m-%d--%H-%M-%S')}-{output_name}-{str(uuid.uuid4())[:8]}"
    )
    return path, base_filename


def save_dataframe(dataframe: pandas.DataFrame, output_path: Union[pathlib.Path, str]):
    """
    Safe the given dataframe as csv at output_path. If the path does not end with the '.csv' suffix, the suffix is
    appended to the filename.

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
    print(f"Saving results to {output_path.absolute()}")
    dataframe["result_filename"] = output_path
    dataframe.to_csv(output_path, index=False)


def mix_colors(
    colors: Collection[Union[str, Collection[float]]],
    ratios: Optional[Collection[float]] = None,
) -> np.array:
    """
    Mix the given colors according to the given ratios (equal ratios if no ratios given) and return the result.

    Parameters
    ----------
    colors : Collection[Union[str, Collection[float]]]
        A collection of the colors to mix. Colors need to be provided in a format parsable by
        'matplotlib.colors.to_rgb', e.g., either a matplotlib color string or an '(a), r, g, b' tuple.
    ratios : Optional[Collection[float]]
        A collection of ratios in which the colors shall be mixed, should have the same size as colors.
        If no ratios are given, equal ratios are used.

    Returns
    -------
    numpy.array
        The mixed color as rgb.
    """
    ratios = np.ones(len(colors)) if ratios is None else np.array(ratios)
    ratios = ratios.reshape(len(colors), 1)
    ratios /= ratios.sum()

    rgb_colors = np.array([matplotlib.colors.to_rgb(color) for color in colors])
    return (ratios * rgb_colors).sum(axis=0)


def mix_with_white(color: Union[str, Collection[float]], color_ratio=0.5) -> np.array:
    """
    Mix the given color with white using the given ratio.

    Parameters
    ----------
    color : Union[str, Collection[float]]
        The color to mix with white, needs to be in a format parsable by 'matplotlib.colors.to_rgb'.
    color_ratio : float
        The ratio of the given color to white (i.e. this color has ratio 'color_ratio', white has ratio
        '1 - color_ratio').

    Returns
    -------
    numpy.array
        The mixed color as rgb.
    """
    white = "#ffffff"
    return mix_colors([color, white], [color_ratio, 1 - color_ratio])


def ticks(
    num_ticks: int,
    max_ticks: int = 10,
    include_upper: bool = False,
    _accumulated_divisor: int = 1,
) -> List[int]:
    """
    Generate an evenly spaced list of ticks using at most max_ticks ticks and a step-size divisible by i * 1e+j for
    i in [1, 2, 5] and j=1, 2,...

    Parameters
    ----------
    num_ticks : int
        The total number of ticks.
    max_ticks : int
        The maximum number of ticks to produce.
    include_upper : bool
        Whether to include the upper limit in the produced ticks.
    _accumulated_divisor : int
        Internal variable to track the current order of the divisor in each recursion, should not be set manually.

    Returns
    -------
    List[int]
        The resulting ticks.
    """
    divisors = [1, 2, 5]
    for divisor in divisors:
        divisor *= _accumulated_divisor
        if num_ticks // divisor <= max_ticks:
            return list(range(0, num_ticks + include_upper * divisor, divisor))
    return ticks(num_ticks, max_ticks, include_upper, _accumulated_divisor * 10)


def get_evenly_spaced_colors(
    num_colors: int, colormap: Union[matplotlib.colors.Colormap, str] = "viridis"
) -> np.array:
    """
    Get evenly spaced colors from the given colormap.

    Parameters
    ----------
    num_colors : int
        The number of colors to pick.
    colormap : Union[plt.Colormap, str]
        The matplotlib colormap to pick from (default: viridis).

    Returns
    -------
    numpy.array
        The selected colors.
    """
    return plt.get_cmap(colormap)(np.linspace(0, 1, num_colors))


def plot_local_class_distributions_as_ridgeplot(
    class_frequencies: np.array,
    y_offset: float = 0.75,
    figsize: Optional[Collection[float]] = None,
    title: str = "",
    class_labels: Optional[Collection] = None,
) -> Tuple[plt.Figure, Collection[plt.Axes]]:
    """
    Plot the local and global class distribution as ridgeplot.

    Parameters
    ----------
    class_frequencies : numpy.array
        The local class frequencies (ranks on axis 0, classes on axis 1).
    y_offset : float
        Offset between the different ridges. The smaller, the more likely the plots are to overlap.
    figsize : Optional[Collection[float]]
        Optional custom figure size. If not specified, the figure size is adjusted using the default width (from
        rcParams) but increasing the height with the number of ridges.
    title : str
        The figure title.
    class_labels : Optional[Collection]
        Optional class labels to use as x-labels. Otherwise, the class indicees are used.

    Returns
    -------
    Tuple[plt.Figure, Collection[plt.Axes]]
        The figure and axis.
    """
    num_ranks, num_classes = class_frequencies.shape
    average_class_frequencies = class_frequencies.mean(axis=0)

    # offset of the baseline for the ridgeplots
    y_offsets = [i * y_offset * class_frequencies.max() for i in range(num_ranks)]
    x_values = range(num_classes)

    colors = get_evenly_spaced_colors(num_ranks)

    height_ratios = [1, num_ranks * 0.5]
    if figsize is None:
        default_width, default_height = matplotlib.rcParams["figure.figsize"]
        figsize = [default_width, default_height * (0.5 + sum(height_ratios) / 25)]

    fig, axes = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": height_ratios}, sharex=True, figsize=figsize
    )
    ax_avg, ax_ridge = axes

    for y, offset, color in reversed(list(zip(class_frequencies, y_offsets, colors))):
        base = np.repeat(offset, len(y))
        ax_ridge.fill_between(
            x_values,
            base,
            base + y,
            color=mix_with_white(color, 0.5),
            zorder=2,
            alpha=0.7,
        )
        ax_ridge.plot(x_values, base + y, color=color, linewidth=0.8)

    yticks = ticks(num_ranks)
    ax_ridge.set_yticks([y_offsets[tick] for tick in yticks], yticks)
    ax_ridge.set_xticks(class_labels if class_labels else ticks(num_classes))
    ax_ridge.set_ylabel("Local Distributions")
    ax_ridge.set_xlabel("Class")

    ax_avg.plot(x_values, average_class_frequencies, color="black", linewidth=0.8)
    ax_avg.set_ylim(0, average_class_frequencies.max() * 1.25)
    ax_avg.set_ylabel("Global")

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_class_distributions(
    class_frequencies_by_label: Dict,
    title: str = "",
    legend_title: str = "",
    labels_order: Optional[List] = None,
) -> Tuple[plt.Figure, Collection[plt.Axes]]:
    """
    Plot different class distributions as line plot.

    Parameters
    ----------
    class_frequencies_by_label : Dict
        The data mapping a label for each line to the corresponding class frequencies.
    title : str
        Optional title for the figure.
    legend_title : str
        Optional title for the labels
    labels_order : Optional[List]
        Optional list of labels specifying the order in which the distributions are plotted. All given labels need to be
        valid keys to class_frequencies_by_label.

    Returns
    -------
    Tuple[plt.Figure, Collection[plt.Axes]]
        The figure and axis.
    """
    labels = labels_order or list(class_frequencies_by_label.keys())
    num_classes = len(class_frequencies_by_label[labels[0]])
    x_values = range(num_classes)

    colors = get_evenly_spaced_colors(len(labels))
    color_map = dict(zip(labels, colors))

    fig, ax = plt.subplots()

    for label in labels:
        ax.plot(
            x_values,
            class_frequencies_by_label[label],
            color=color_map[label],
            label=f"{label:g}" if isinstance(label, float) else label,
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Class weight")
    ax.set_xlabel("Class")
    ax.set_xticks(ticks(num_classes))

    fig.suptitle(title)
    ax.legend(title=legend_title, frameon=False)
    fig.tight_layout()
    return fig, ax


def dataframe_from_slurm_output(path: Union[pathlib.Path, str]) -> pandas.DataFrame:
    """
    Create a dataframe from SLURM output files and save it to csv file.

    Parameters
    ----------
    path : pathlib.Path | str
        Path to folder with SLURM output files

    Returns
    -------
    pandas.DataFrame
        Dataframe with (result) parameters of provided SLURM jobs
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
        wall-clock time string

    Returns
    -------
    Union[float, None]
        wall-clock time in seconds (``None`` if provided string was invalid)
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
        return None  # Return ``None`` for invalid time strings


def plot_single_node_capacity(
    df: pandas.DataFrame, save_fig: Union[pathlib.Path, str] = pathlib.Path("./")
) -> None:
    """
    Plot results from single-node capacity experiments.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with (result) parameters of provided SLURM jobs
    save_fig : Union[pathlib.Path, str]
        Path to save figure to
    """
    # Define marker colors based on job state.
    marker_colors = {
        "COMPLETED": "C2",  # green
        "FAILED": "C3",  # red
        "TIMEOUT": "C1",  # orange
    }

    df["marker_color"] = df["job_state"].map(
        marker_colors
    )  # Map job states to marker colors.

    # For each experiment, plot number of features vs. number of samples. Marker size corresponds to number of trees
    # trained. Marker color indicates job state, i.e., green for COMPLETED, orange for TIMEOUT, and red for FAILED.
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.scatter(
        x=df["n_samples"].astype(float),
        y=df["n_features"].astype(float),
        s=2 * np.log10(df["n_trees"].astype(float)) ** 4.85,
        c=df["marker_color"],
        edgecolor="face",
        alpha=0.35,
        zorder=10,
        clip_on=True,
    )
    ax.set_ylabel("Number of features", weight="bold")
    ax.set_xlabel("Number of samples", weight="bold")
    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
    # ax.set_title("Single-node capacity", weight="normal", size=14)

    # Add descriptive text.
    ax.text(
        1.02,
        0.5,
        "COMPLETE",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        color="C2",
        weight="bold",
    )
    ax.text(
        1.02,
        0.45,
        "TIMEOUT",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        color="C1",
        weight="bold",
    )
    ax.text(
        1.02,
        0.4,
        "OUT OF MEMORY",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        color="C3",
        weight="bold",
    )

    ax.text(
        0.1,
        -0.2,
        "Markers grow with number of trees as "
        + r"$\left[10^2, 10^3, 10^4, 10^5\right]$"
        + ".",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        fontstyle="italic",
    )

    ax.text(
        -0.05,
        -0.24,
        "Jobs failed with numpy.core._exceptions.MemoryError during dataset creation.",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        fontstyle="italic",
    )

    plt.tight_layout()
    plt.savefig(save_fig / pathlib.Path("single_node_capacity.pdf"))
    plt.show()


def plot_times_and_accuracy(
    df: pandas.DataFrame, save_fig: Union[pathlib.Path, str] = pathlib.Path("./")
) -> None:
    """
    Plot accuracy from single-node capacity experiments.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with (result) parameters of provided SLURM jobs
    save_fig : Union[pathlib.Path, str]
        Path to save figure to
    """
    df_filtered = df[
        df["job_state"] == "COMPLETED"
    ]  # Filter dataframe for rows with job state "COMPLETED".
    datasets = df_filtered[
        ["n_samples", "n_features"]
    ].drop_duplicates()  # Get unique dataset combinations.

    # Plot training time and accuracy of completed jobs (top) as well as wall-clock time of jobs that did not fail
    # (bottom) vs. number of trees.
    fig, (axt, axb) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7, 7))
    axt.set_ylabel("Training time / min", weight="bold")
    axt.set_xscale("log", base=10)
    axt.set_yscale("log", base=10)
    axacc = axt.twinx()
    axacc.set_ylabel("Accuracy / %", weight="bold")

    for i, dataset in datasets.iterrows():
        n_samples = dataset["n_samples"]
        n_features = dataset["n_features"]
        dataset_df = df_filtered[
            (df_filtered["n_samples"] == n_samples)
            & (df_filtered["n_features"] == n_features)
        ]  # Filter dataframe for current dataset.

        # Plot training time and accuracy vs. number of trees for the current dataset.
        axt.plot(
            dataset_df["n_trees"].astype(float),
            dataset_df["training_time"] / 60,
            ms=10,
            linestyle="-",
            marker="o",
            label=f"{n_samples} samples, {n_features} features",
            alpha=0.35,
        )
        axacc.plot(
            dataset_df["n_trees"].astype(float),
            dataset_df["accuracy"].astype(float) * 100,
            ms=10,
            linestyle="-",
            marker="X",
        )
    axt.grid(True)
    axt.legend(loc="lower right", fontsize=7)
    axacc.text(
        1.2,
        0.4,
        "Circles show training\ntimes, crosses show\naccuracies.",
        transform=axacc.transAxes,
        fontsize=8,
        verticalalignment="center",
        fontstyle="italic",
    )

    datasets = df[
        ["n_samples", "n_features"]
    ].drop_duplicates()  # Get unique dataset combinations.
    axb.set_xlabel("Number of trees", weight="bold")
    axb.set_ylabel("Wall-clock time / min", weight="bold")
    axb.set_xscale("log", base=10)
    axb.set_yscale("log", base=10)
    markers = [
        "o",
        "v",
        "*",
        ">",
        "<",
        "d",
        "^",
        "p",
        "P",
        "8",
        "h",
        "X",
        "D",
        "s",
        "1",
        "2",
    ]

    legend_entries = []

    for i, dataset in datasets.iterrows():
        n_samples = dataset["n_samples"]
        n_features = dataset["n_features"]

        label = f"{n_samples} samples, {n_features} features"
        legend_entries.append(
            Line2D(
                [0],
                [0],
                marker=markers[i],
                color="k",
                label=label,
                markersize=6,
                alpha=0.35,
            ),
        )
        # Filter dataframe for current dataset
        dataset_df = df[
            (df["n_samples"] == n_samples) & (df["n_features"] == n_features)
        ]
        axb.scatter(
            dataset_df["n_trees"].astype(float),
            dataset_df["wall_clock_time"].apply(time_to_seconds) / 60,
            s=80,
            marker=markers[i],
            c=dataset_df["marker_color"],
            alpha=0.5,
        )
        axb.plot(
            dataset_df["n_trees"].astype(float),
            dataset_df["wall_clock_time"].apply(time_to_seconds) / 60,
            linestyle="dotted",
            alpha=0.35,
            c="k",
        )
    axb.grid(True)
    box = axb.get_position()
    axb.set_position(
        [box.x0, box.y0, box.width * 0.8, box.height]
    )  # Shrink current axis by 20 %.
    axb.legend(
        handles=legend_entries, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7
    )  # Put legend to the right of the current axis.
    plt.tight_layout()
    plt.savefig(save_fig / pathlib.Path("single_node_times_acc.pdf"))
    plt.show()

if __name__ == "__main__":
    plot_single_node_capacity()