import argparse
import logging
import pathlib

from specialcouscous.utils import set_logger_config
from specialcouscous.utils.train import train_serial_on_synthetic_data

log = logging.getLogger("specialcouscous")  # Get logger instance.

if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        prog="Random Forest",
        description="Generate synthetic classification data and classify with random forest.",
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
        default=10,
        help="Number of classes in synthetic classification data",
    )
    parser.add_argument(
        "--n_clusters_per_class",
        type=int,
        default=1,
        help="Number of clusters per class",
    )
    parser.add_argument(
        "--random_state_data",
        type=int,
        default=0,
        help="Random seed used in synthetic dataset generation",
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
        "--random_state_split",
        type=int,
        default=9,
        help="Random seed used in train-test split",
    )
    # Model-related arguments
    parser.add_argument(
        "--n_trees",
        type=int,
        default=100,
        help="Number of trees in global random forest classifier",
    )
    parser.add_argument(
        "--random_state_forest",
        type=int,
        default=0,
        help="Random seed used to initialize random forest classifier",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.75,
        help="Fraction of data in the train set. The remainder makes up the test set.",
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
        default="serial_rf",
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
    args = parser.parse_args()

    # Set up separate logger for Special Couscous.
    set_logger_config(
        level=args.logging_level,  # Logging level
        log_file=f"{args.log_path}/{pathlib.Path(__file__).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    log.info(
        "**************************************************************\n"
        "* Single-Node Random Forest Classification of Synthetic Data *\n"
        "**************************************************************\n"
        f"Hyperparameters used are:\n{args}"
    )

    train_serial_on_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_clusters_per_class=args.n_clusters_per_class,
        frac_informative=args.frac_informative,
        frac_redundant=args.frac_redundant,
        seed_data=args.random_state_data,
        seed_split=args.random_state_split,
        seed_model=args.random_state_forest,
        train_split=args.train_split,
        n_trees=args.n_trees,
        output_dir=args.output_dir,
        output_label=args.output_label,
        experiment_id=args.experiment_id,
        save_model=args.save_model,
    )