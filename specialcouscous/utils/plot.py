import pathlib
from typing import Collection

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.lines import Line2D

from specialcouscous.utils.slurm import time_to_seconds

AVERAGE_KWARGS = {
    "marker": "x",
    "s": 38,
    "color": "C0",
    "linewidths": 1,
    "zorder": 10,
}
INDIVIDUAL_KWARGS = {
    "marker": ",",
    "color": "k",
    "s": 5,
    "alpha": 0.3,
    "zorder": 20,
}
ERROR_KWARGS = {
    "fmt": "_",
    "color": "C2",
    "ecolor": "C2",
    "elinewidth": 1,
    "linewidth": 1,
    "capsize": 0.9,
    "ms": 5,
    "zorder": 10,
}
GLOBAL_ERROR_KWARGS = {
    "fmt": "_",
    "color": "C0",
    "ecolor": "C0",
    "elinewidth": 1,
    "linewidth": 1,
    "capsize": 0.9,
    "ms": 5,
    "zorder": 20,
}
LINE_KWARGS = {
    "linestyle": "--",
    "color": "C2",
    "linewidth": 0.5,
    "alpha": 0.5,
    "zorder": 10,
}
GLOBAL_LINE_KWARGS = {
    "linestyle": "dashed",
    "color": "C0",
    "linewidth": 0.5,
    "alpha": 0.5,
    "zorder": 10,
}
DATA_SEED_KWARGS = {
    "marker": "1",
    "s": 38,
    "color": "C0",
    "linewidths": 1,
}


def mix_colors(
    colors: Collection[str | Collection[float]],
    ratios: Collection[float] | None = None,
) -> np.ndarray:
    """
    Mix the given colors according to the given ratios (equal ratios if no ratios given) and return the result.

    Parameters
    ----------
    colors : Collection[str | Collection[float]]
        A collection of the colors to mix. Colors need to be provided in a format parsable by
        ``matplotlib.colors.to_rgb``, e.g., either a matplotlib color string or an '(a), r, g, b' tuple.
    ratios : Collection[float], optional
        A collection of ratios in which the colors shall be mixed, should have the same size as colors.
        If no ratios are given, equal ratios are used.

    Returns
    -------
    numpy.ndarray
        The mixed color as rgb.
    """
    ratios_array = np.ones(len(colors)) if ratios is None else np.array(ratios)
    ratios_array = ratios_array.reshape(len(colors), 1)
    ratios_array /= ratios_array.sum()

    rgb_colors = np.array([matplotlib.colors.to_rgb(color) for color in colors])
    return (ratios_array * rgb_colors).sum(axis=0)


def mix_with_white(
    color: str | Collection[float], color_ratio: float = 0.5
) -> np.ndarray:
    """
    Mix the given color with white using the given ratio.

    Parameters
    ----------
    color : str | Collection[float]
        The color to mix with white, needs to be in a format parsable by ``matplotlib.colors.to_rgb``.
    color_ratio : float
        The ratio of the given color to white (i.e. this color has ratio ``color_ratio``, white has ratio
        '1 - ``color_ratio``').

    Returns
    -------
    numpy.ndarray
        The mixed color as rgb.
    """
    white = "#ffffff"
    return mix_colors([color, white], [color_ratio, 1 - color_ratio])


def ticks(
    num_ticks: int,
    max_ticks: int = 10,
    include_upper: bool = False,
    _accumulated_divisor: int = 1,
) -> list[int]:
    """
    Generate an evenly spaced list of ticks.

    Use at most ``max_ticks`` ticks and a step-size divisible by i * 1e+j for i in [1, 2, 5] and j=1, 2,...

    Parameters
    ----------
    num_ticks : int
        The total number of ticks.
    max_ticks : int
        The maximum number of ticks to produce.
    include_upper : bool
        Whether to include the upper limit in the produced ticks. Default is False.
    _accumulated_divisor : int
        Internal variable to track the current order of the divisor in each recursion, should not be set manually.

    Returns
    -------
    list[int]
        The resulting ticks.
    """
    divisors = [1, 2, 5]
    for divisor in divisors:
        divisor *= _accumulated_divisor
        if num_ticks // divisor <= max_ticks:
            return list(range(0, num_ticks + include_upper * divisor, divisor))
    return ticks(num_ticks, max_ticks, include_upper, _accumulated_divisor * 10)


def get_evenly_spaced_colors(
    num_colors: int, colormap: matplotlib.colors.Colormap | str = "viridis"
) -> np.ndarray:
    """
    Get evenly spaced colors from the given colormap.

    Parameters
    ----------
    num_colors : int
        The number of colors to pick.
    colormap : plt.Colormap | str
        The matplotlib colormap to pick from. Default is viridis.

    Returns
    -------
    numpy.ndarray
        The selected colors.
    """
    return plt.get_cmap(colormap)(np.linspace(0, 1, num_colors))


def plot_local_class_distributions_as_ridgeplot(
    class_frequencies: np.ndarray,
    y_offset: float = 0.75,
    figsize: Collection[float] | None = None,
    title: str = "",
    class_labels: Collection | None = None,
) -> tuple[plt.Figure, Collection[plt.Axes]]:
    """
    Plot the local and global class distribution as ridgeplot.

    Parameters
    ----------
    class_frequencies : numpy.ndarray
        The local class frequencies (ranks on axis 0, classes on axis 1).
    y_offset : float
        Offset between the different ridges. The smaller, the more likely the plots are to overlap.
    figsize : Collection[float], optional
        Optional custom figure size. If not specified, the figure size is adjusted using the default width (from
        ``rcParams``) but increasing the height with the number of ridges.
    title : str
        The figure title.
    class_labels : Collection, optional
        Optional class labels to use as x-labels. Otherwise, the class indices are used.

    Returns
    -------
    tuple[plt.Figure, Collection[plt.Axes]]
        The figure and axis.
    """
    num_ranks, num_classes = class_frequencies.shape
    average_class_frequencies = class_frequencies.mean(axis=0)

    # Offset of the baseline for the ridge plots
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
    ax_ridge.set_ylabel("Local distributions")
    ax_ridge.set_xlabel("Class")

    ax_avg.plot(x_values, average_class_frequencies, color="black", linewidth=0.8)
    ax_avg.set_ylim(0, average_class_frequencies.max() * 1.25)
    ax_avg.set_ylabel("Global")

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_class_distributions(
    class_frequencies_by_label: dict,
    title: str = "",
    legend_title: str = "",
    labels_order: list | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot different class distributions as line plot.

    Parameters
    ----------
    class_frequencies_by_label : dict
        The data mapping a label for each line to the corresponding class frequencies.
    title : str
        Optional title for the figure.
    legend_title : str
        Optional title for the labels
    labels_order : list | None
        Optional list of labels specifying the order in which the distributions are plotted. All given labels need to be
        valid keys to ``class_frequencies_by_label``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
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


def plot_single_node_capacity(
    df: pandas.DataFrame, save_fig: pathlib.Path | str = pathlib.Path("../")
) -> None:
    """
    Plot results from single-node capacity experiments.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe with (result) parameters of the provided SLURM jobs.
    save_fig : pathlib.Path | str
        The path to save the figure to.
    """
    # Define marker colors based on job state.
    marker_colors = {
        "COMPLETED": "C2",  # green
        "OUT OF MEMORY": "C3",  # red
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
    ax.set_title("Single-node capacity", weight="normal", size=14)

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
    df: pandas.DataFrame, save_fig: pathlib.Path | str = pathlib.Path("../")
) -> None:
    """
    Plot training time and accuracy from single-node capacity experiments.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe with (result) parameters of the provided SLURM jobs.
    save_fig : pathlib.Path | str
        The path to save the figure to.
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
    ax_acc = axt.twinx()
    ax_acc.set_ylabel("Accuracy / %", weight="bold")

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
        ax_acc.plot(
            dataset_df["n_trees"].astype(float),
            dataset_df["accuracy"].astype(float) * 100,
            ms=10,
            linestyle="-",
            marker="X",
        )
    axt.grid(True)
    axt.legend(loc="lower right", fontsize=7)
    ax_acc.text(
        1.2,
        0.4,
        "Circles show training\ntimes, crosses show\naccuracies.",
        transform=ax_acc.transAxes,
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
