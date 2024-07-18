import logging
import pathlib
import shutil

import pytest

from specialcouscous.utils import set_logger_config
from specialcouscous.utils.train import train_serial_on_synthetic_data

log = logging.getLogger("specialcouscous")  # Get logger instance.


@pytest.mark.mpi_skip
def test_serial_synthetic() -> None:
    """Test serial training of random forest on synthetic data."""
    # Data-related arguments
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
    detailed_evaluation: bool = True  # Whether to perform a detailed evaluation on more than just the local test set.
    output_dir: pathlib.Path = pathlib.Path(
        "./results"
    )  # Directory to write results to
    experiment_id: str = (
        "test_serial_rf"  # Optional subdirectory name to collect related result in
    )
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

    log.info(
        "**************************************************************\n"
        "* Single-Node Random Forest Classification of Synthetic Data *\n"
        "**************************************************************"
    )

    train_serial_on_synthetic_data(
        n_samples,
        n_features,
        n_classes,
        n_clusters_per_class,
        frac_informative,
        frac_redundant,
        n_trees=n_trees,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        experiment_id=experiment_id,
    )
    log_file.unlink()
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    test_serial_synthetic()
