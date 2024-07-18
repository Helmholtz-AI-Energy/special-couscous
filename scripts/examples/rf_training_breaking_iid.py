import argparse
import logging
import pathlib

from mpi4py import MPI

from specialcouscous.utils import set_logger_config
from specialcouscous.utils.train import train_parallel_on_synthetic_data

log = logging.getLogger("specialcouscous")  # Get logger instance.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Distributed Random Forests on Synthetic Classification Data"
    )
    parser.add_argument("--random_state_data", type=int, default=0)
    parser.add_argument("--random_state_forest", type=int, default=0)

    parser.add_argument("--n_trees", type=int, default=100)
    parser.add_argument("--global_model", action="store_true")

    parser.add_argument(
        "--n_classes",
        type=int,
        default=5,
        help="Number of classes in synthetic classification data",
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
        "--private_test_set",
        action="store_true",
        help="Whether the test set is private (not shared across subforests)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data in the train set. The remainder makes up the test set.",
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
        default="parallel_rf_breaking_iid",
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

    train_parallel_on_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        globally_balanced=not args.globally_imbalanced,
        locally_balanced=not args.locally_imbalanced,
        shared_test_set=not args.private_test_set,
        seed_data=args.random_state_data,
        seed_model=args.random_state_forest,
        mu_partition=args.mu_partition,
        mu_data=args.mu_data,
        peak=args.peak,
        comm=MPI.COMM_WORLD,
        train_split=args.train_split,
        n_trees=args.n_trees,
        global_model=args.global_model,
        detailed_evaluation=args.detailed_evaluation,
        output_dir=args.output_dir,
        output_label=args.output_label,
        experiment_id=args.experiment_id,
    )
