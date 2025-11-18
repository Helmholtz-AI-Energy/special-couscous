import argparse
import logging
import pathlib
from typing import Any


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
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["susy", "cover_type", "higgs"],
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
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of parallel jobs per rank. Default is -1 to use all available cores. "
        "Adjust when running multiple ranks on one node.",
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
        "--distribute_data",
        action="store_true",
        help="Whether to distribute the data across nodes. Currently only supported for synthetic data.",
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


def get_synthetic_data_kwargs(cli_args: argparse.Namespace) -> dict[str, Any]:
    """
    Get parameters for the synthetic dataset from CLI arguments and return as dict.

    Parameters
    ----------
    cli_args : argparse.Namespace
        The CLI arguments.

    Returns
    -------
    dict[str, Any]
        The parameters for the synthetic dataset.
    """
    return {
        "n_samples": cli_args.n_samples,
        "n_features": cli_args.n_features,
        "n_classes": cli_args.n_classes,
        "make_classification_kwargs": {
            "n_clusters_per_class": cli_args.n_clusters_per_class,
            "n_informative": int(cli_args.frac_informative * cli_args.n_features),
            "n_redundant": int(cli_args.frac_redundant * cli_args.n_features),
            "flip_y": cli_args.flip_y,
        },
    }


def get_general_run_kwargs(cli_args: argparse.Namespace) -> dict[str, Any]:
    """
    Get general run parameters from CLI arguments and return as dict.

    Parameters
    ----------
    cli_args : argparse.Namespace
        The CLI arguments.

    Returns
    -------
    dict[str, Any]
        The general parameters used by all scripts.
    """
    return {
        "random_state": cli_args.random_state,
        "random_state_model": cli_args.random_state_model,
        "train_split": cli_args.train_split,
        "stratified_train_test": cli_args.stratified_train_test,
        "n_trees": cli_args.n_trees,
        "detailed_evaluation": cli_args.detailed_evaluation,
        "output_dir": cli_args.output_dir,
        "output_label": cli_args.output_label,
        "experiment_id": cli_args.experiment_id,
        "save_model": cli_args.save_model,
    }
