import logging
import pathlib
import shutil

import pytest
from mpi4py import MPI

from specialcouscous.train import train_parallel_on_balanced_synthetic_data
from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


@pytest.mark.mpi
def test_parallel_synthetic() -> None:
    """Test parallel training of random forest on synthetic data."""
    n_samples: int = 1000  # Number of samples in synthetic classification data
    n_features: int = 100  # Number of features in synthetic classification data
    n_classes: int = 10  # Number of classes in synthetic classification data
    n_clusters_per_class: int = 1  # Number of clusters per class
    frac_informative: float = (
        0.1  # Fraction of informative features in synthetic classification dataset
    )
    frac_redundant: float = (
        0.1  # Fraction of redundant features in synthetic classification dataset
    )
    # Model-related arguments
    n_trees: int = 100  # Number of trees in global random forest classifier
    output_dir: pathlib.Path = pathlib.Path(
        "./results"
    )  # Directory to write results to
    experiment_id: str = (
        "test_parallel_rf"  # Optional subdirectory name to collect related result in
    )
    save_model: bool = True
    global_model: bool = True
    detailed_evaluation: bool = True  # Whether to perform a detailed evaluation on more than just the local test set.
    log_path: pathlib.Path = pathlib.Path("./")  # Path to the log directory
    logging_level: int = logging.INFO  # Logging level
    log_file: pathlib.Path = pathlib.Path(
        f"{log_path}/{pathlib.Path(__file__).stem}.log"
    )

    # Set up separate logger for Special Couscous.
    set_logger_config(
        level=logging_level,  # Logging level
        log_file=log_file,  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        log.info(
            "*************************************************************\n"
            "* Multi-Node Random Forest Classification of Synthetic Data *\n"
            "*************************************************************"
        )

    train_parallel_on_balanced_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        frac_informative=frac_informative,
        frac_redundant=frac_redundant,
        mpi_comm=comm,
        n_trees=n_trees,
        global_model=global_model,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        experiment_id=experiment_id,
        save_model=save_model,
    )
    if comm.rank == 0:
        log_file.unlink()
        shutil.rmtree(output_dir)
