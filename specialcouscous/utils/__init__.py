import io
import logging
import pathlib
import pickle
import sys
import typing

import colorlog
import numpy as np
from mpi4py import MPI

from .parse_cli_args import parse_arguments  # noqa: F401

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
