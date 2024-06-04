import argparse
import pathlib
import re
from typing import Union, Optional

import numpy as np
import pandas
from mpi4py import MPI

import RF_parallel
import utils
from synthetic_classification_data import make_classification_dataset, SyntheticDataset


def train_parallel_on_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_clusters_per_class: int,
    frac_informative: float,
    frac_redundant: float,
    seed_data: int = 0,
    seed_split: int = 0,
    seed_model: int = 0,
    comm: MPI.Comm = MPI.COMM_WORLD,
    train_split: float = 0.75,
    n_trees: int = 100,
    global_model: bool = True,
    detailed_evaluation: bool = False,
    output_dir: Optional[Union[pathlib.Path, str]] = None,
    output_label: str = "",
    experiment_id: Optional[str] = None,
) -> None:
    """
    Train and evaluate a distributed random forest on synthetic data.

    Parameters
    ----------
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number of classes in the dataset.
    n_clusters_per_class : int
        The number of clusters per class in the dataset.
    frac_informative : float
        The fraction of informative features in the dataset.
    frac_redundant : float
        The fraction of redundant features in the dataset.
    seed_data : int
        The random seed used for the dataset generation.
    seed_split : int
        The random seed used to train-test split the data.
    seed_model : int
        The random seed used for the model.
    comm : MPI.Comm
        The MPI communicator to distribute over.
    train_split : float
        Relative size of the train set.
    n_trees : int
        The number of trees in the global forest.
    global_model : bool
        Whether the local models are all-gathered to one global model shared by all ranks after training.
    detailed_evaluation : bool
        Whether to perform a detailed evaluation on more than just the local test set.
    output_dir : Optional[Union[pathlib.Path, str]]
        Output base directory. If given, the results are written to
        output_dir / year / year-month / date / YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>.
    output_label : str
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment.
    experiment_id : Optional[str]
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment.
    """
    # get all arguments passed to the function as dict, captures all variables in the current local scope so this needs
    # to be called before defining any other local variables
    configuration = locals()
    for key in ["comm", "output_dir", "detailed_evaluation"]:
        del configuration[key]
    configuration["comm_size"] = comm.size

    global_results = {"comm_rank": "global"}
    local_results = {"comm_rank": comm.rank}

    def store_timing(timer):
        label = "time_sec_" + re.sub(r"\s", "_", timer.name)
        global_results[label] = timer.elapsed_time_average
        local_results[label] = timer.elapsed_time_local

    def store_accuracy(model, label):
        global_results[f"accuracy_{label}"] = model.acc_global
        local_results[f"accuracy_{label}"] = model.acc_local

    # -------------- Generate and distribute data --------------
    if comm.rank == 0:
        print("Generating synthetic data.")
    with utils.MPITimer(comm, name="data generation") as timer:
        (
            train_samples,
            test_samples,
            train_targets,
            test_targets,
        ) = make_classification_dataset(
            n_samples=n_samples,
            n_features=n_features,
            frac_informative=frac_informative,
            frac_redundant=frac_redundant,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            random_state_generation=seed_data,
            train_split=train_split,
            random_state_split=seed_split,
        )

        train_data = SyntheticDataset(x=train_samples, y=train_targets)
        test_data = SyntheticDataset(x=test_samples, y=test_targets)

    store_timing(timer)

    print(
        f"Done\nTrain samples and targets have shapes {train_data.x.shape} and {train_data.y.shape}.\n"
        # f"First three elements are: {train_data.x[:3]} and {train_data.y[:3]}\n"
        f"Test samples and targets have shapes {test_data.x.shape} and {test_data.y.shape}.\n"
        # f"First three elements are: {test_data.x[:3]} and {test_data.y[:3]}\n"
        f"Set up classifier."
    )

    # -------------- Setup and train random forest --------------
    print(f"[{comm.rank}/{comm.size}]: Set up and train local random forest.")
    with utils.MPITimer(comm, name="forest creation") as timer:
        distributed_random_forest = RF_parallel.DistributedRandomForest(
            n_trees_global=n_trees,
            comm=comm,
            random_state=seed_model,
            global_model=global_model,
        )
    store_timing(timer)

    with utils.MPITimer(comm, name="training") as timer:
        if global_model:
            timer_sync_global_model = distributed_random_forest.train(
                train_data.x, train_data.y, global_model
            )
            store_timing(timer_sync_global_model)
        else:
            distributed_random_forest.train(train_data.x, train_data.y, global_model)
    store_timing(timer)

    # -------------- Evaluate random forest --------------
    print(f"[{comm.rank}/{comm.size}]: Evaluate random forest.")
    with utils.MPITimer(comm, name="test") as timer:  # Test trained model on test data.
        distributed_random_forest.test(
            test_data.x, test_data.y, n_classes, global_model
        )
    store_timing(timer)
    store_accuracy(distributed_random_forest, "test")

    if detailed_evaluation:  # Test trained model also on training data.
        distributed_random_forest.test(
            train_samples, train_targets, n_classes, global_model
        )
        store_accuracy(distributed_random_forest, "train")

    # -------------- Gather local results, generate dataframe, output collective results --------------
    # Convert local results to array, ensure the values are in the same order on all ranks by sorting the keys.
    key_order = sorted(local_results.keys())
    local_results_array = np.array([local_results[key] for key in key_order])

    gathered_local_results = comm.gather(local_results_array)
    gathered_class_frequencies_train = train_data.gather_class_frequencies(comm)
    gathered_class_frequencies_test = test_data.gather_class_frequencies(comm)
    if comm.rank == 0:
        # Convert arrays back into dicts, then into dataframe.
        gathered_local_results = [
            dict(zip(key_order, gathered_values))
            for gathered_values in gathered_local_results
        ]
        results_df = pandas.DataFrame(gathered_local_results + [global_results])
        # Add configuration as columns.
        for key, value in configuration.items():
            results_df[key] = value

        if output_dir:
            path = pathlib.Path(output_dir)
            _, base_filename = utils.construct_output_path(
                output_dir, output_label, experiment_id
            )
            utils.save_dataframe(results_df, path / (base_filename + "_results.csv"))
            (
                fig_train,
                _,
            ) = SyntheticDataset.plot_local_class_distributions(
                gathered_class_frequencies_train
            )
            (
                fig_test,
                _,
            ) = SyntheticDataset.plot_local_class_distributions(
                gathered_class_frequencies_test
            )
            fig_train.savefig(path / (base_filename + "_class_distribution_train.pdf"))
            fig_test.savefig(path / (base_filename + "_class_distribution_test.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Distributed Random Forests on Synthetic Classification Data"
    )
    # Data-related arguments
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
    parser.add_argument("--global_model", action="store_true")

    parser.add_argument(
        "--train_split",
        type=float,
        default=0.75,
        help="Fraction of data in the train set. The remainder makes up the test set.",
    )

    parser.add_argument(
        "--detailed_evaluation",
        action="store_true",
        help="Whether to perform a detailed evaluation on more than just the local test set.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent / "results",
        help="The directory to write the results to.",
    )
    parser.add_argument(
        "--output_label",
        type=str,
        default="parallel_rf",
        help="Optional label for the output files.",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        help="Optional subdirectory name to collect related "
        "result in. The subdirectory will be created automatically.",
    )

    comm = MPI.COMM_WORLD
    args = parser.parse_args()
    if comm.rank == 0:
        print(
            "*************************************************************\n"
            "* Multi-Node Random Forest Classification of Synthetic Data *\n"
            "*************************************************************\n"
            f"Hyperparameters used are:\n{args}"
        )

    train_parallel_on_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_clusters_per_class=args.n_clusters_per_class,
        frac_informative=args.frac_informative,
        frac_redundant=args.frac_redundant,
        seed_data=args.random_state_data,
        seed_model=args.random_state_forest,
        seed_split=args.random_state_split,
        comm=comm,
        train_split=args.train_split,
        n_trees=args.n_trees,
        global_model=args.global_model,
        detailed_evaluation=args.detailed_evaluation,
        output_dir=args.output_dir,
        output_label=args.output_label,
        experiment_id=args.experiment_id,
    )
