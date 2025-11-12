import argparse
import io
import logging
import pathlib
import pickle
import sys
import typing

import colorlog
import numpy as np
from mpi4py import MPI

import specialcouscous.utils.datasets

log = logging.getLogger(__name__)  # Get logger instance.


def get_pickled_size(x: object, **kwargs: typing.Any) -> int:
    """
    Get the size in bytes of the given object x after pickling.

    This can for example be used to determine the message size of objects send via MPI.

    Parameters
    ----------
    x : object
        The object whose size to get.
    kwargs : Any
        Additional keywords passed through to pickle.dump.

    Returns
    -------
    int
        The size of the object x (after pickling) in bytes.
    """
    binary_representation = io.BytesIO()
    pickle.dump(x, binary_representation, **kwargs)
    return sys.getsizeof(binary_representation)


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
    log_file: str | pathlib.Path | None = None,
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


def parse_arguments() -> argparse.Namespace:
    """
    Set up argument parser for random forest classification in ``special-couscous``.

    Returns
    -------
    argparse.Namespace
        The namespace of all parsed arguments.
    """
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        prog="Random Forest",
        description="Generate synthetic classification data and classify with (distributed) random forest.",
    )
    available_datasets = list(specialcouscous.utils.datasets.DATASETS.keys())
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=available_datasets[0],
        choices=available_datasets,
        help="The dataset to train on when not using a synthetic dataset.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples in synthetic classification data",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=100,
        help="Number of features in synthetic classification data",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=5,
        help="Number of classes in synthetic classification data",
    )
    parser.add_argument(
        "--n_clusters_per_class",
        type=int,
        default=1,
        help="Number of clusters per class",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random state used for synthetic dataset generation and model instantiation.",
    )
    parser.add_argument(
        "--frac_informative",
        type=float,
        default=0.1,
        help="Fraction of informative features in synthetic classification dataset",
    )
    parser.add_argument(
        "--frac_redundant",
        type=float,
        default=0.1,
        help="Fraction of redundant features in synthetic classification dataset",
    )
    parser.add_argument(
        "--flip_y",
        type=float,
        default=0.01,
        help="The fraction of samples whose class is assigned randomly. Larger values introduce noise in the labels "
        "and make the classification task harder. Note that the default setting `flip_y` > 0 might lead to less "
        "than `n_classes` in `y` in some cases.",
    )
    parser.add_argument(
        "--stratified_train_test",
        action="store_true",
        help="Whether to stratify the train-test split with the class labels.",
    )
    # Model-related arguments
    parser.add_argument(
        "--n_trees",
        type=int,
        default=100,
        help="Number of trees in global random forest classifier",
    )
    parser.add_argument(
        "--random_state_model",
        type=int,
        default=None,
        help="Optional random seed used to initialize the random forest classifier",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.75,
        help="Fraction of data in the train set. The remainder makes up the test set.",
    )
    parser.add_argument(
        "--shared_global_model",
        action="store_true",
        help="Whether the local classifiers are all-gathered so that each rank can access the shared global model",
    )
    parser.add_argument(
        "--shared_test_set",
        action="store_true",
        help="Whether the test set is shared across all subforests",
    )
    parser.add_argument(
        "--globally_imbalanced",
        action="store_true",
        help="Whether the global dataset has class imbalance. If true, the classes are Skellam "
        "distributed. Use --mu_data and --peak to customize the distribution.",
    )
    parser.add_argument(
        "--mu_data",
        type=float,
        default=10,
        help="The μ = μ₁ = μ₂, μ ∊ [0, ∞] parameter of the Skellam distribution. The larger μ, the "
        "larger the spread. Edge cases: For μ=0, the peak class has weight 1, while all other "
        "classes have weight 0. For μ=inf, the generated dataset is balanced, i.e., all classes "
        "have equal weights.",
    )
    parser.add_argument(
        "--peak",
        type=int,
        default=0,
        help="The position (class index) of the distribution's peak (i.e., the most frequent class).",
    )
    parser.add_argument(
        "--enforce_constant_local_size",
        action="store_true",
        help="Whether to relax the local class distribution to instead force all local subsets to have the same size.",
    )
    parser.add_argument(
        "--locally_imbalanced",
        action="store_true",
        help="Whether the partition to local datasets has class imbalance. If true, the classes are "
        "Skellam distributed. Use --mu_partition to customize spread of the distributions.",
    )
    parser.add_argument(
        "--mu_partition",
        type=float,
        default=10.0,
        help="The μ = μ₁ = μ₂, μ ∊ [0, ∞] parameter of the Skellam distribution. The larger μ, the "
        "larger the spread. Edge cases: For μ=0, the peak class has weight 1, while all other "
        "classes have weight 0. For μ=inf, the generated dataset is balanced, i.e., all classes "
        "have equal weights.",
    )
    parser.add_argument(
        "--detailed_evaluation",
        action="store_true",
        help="Whether to perform a detailed evaluation on more than just the local test set.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent.parent / "results",
        help="The directory to write the results to.",
    )
    parser.add_argument(
        "--output_label",
        type=str,
        default="",
        help="Optional label for the output files.",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default="",
        help="Optional subdirectory name to collect related "
        "result in. The subdirectory will be created automatically.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Whether to save the trained classifier to disk.",
    )
    parser.add_argument(
        "--log_path",
        type=pathlib.Path,
        default=pathlib.Path("./"),
        help="Path to the log directory. The directory will be created automatically if it does not exist.",
    )
    parser.add_argument(
        "--logging_level",
        type=int,
        default=logging.INFO,
        help="Logging level.",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=pathlib.Path,
        default=pathlib.Path("./"),
        help="Path to the checkpoint directory containing the local model pickle files to load.",
    )

    parser.add_argument(
        "--checkpoint_uid",
        type=str,
        default="",
        help="The considered run's unique identifier. Used to identify the correct checkpoints to load.",
    )

    return parser.parse_args()
