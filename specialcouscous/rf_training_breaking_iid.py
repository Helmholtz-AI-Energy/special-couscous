import argparse
import pathlib
import re
from typing import Union, Optional

import numpy as np
import pandas
from mpi4py import MPI

import RF_parallel
import synthetic_classification_data
import utils


def train_parallel_on_synthetic_data(
    globally_balanced: bool,
    locally_balanced: bool,
    shared_test_set: bool,
    num_samples: int,
    num_classes: int,
    seed_data: int = 0,
    seed_model: int = 0,
    mu_partition: Optional[Union[float, str]] = None,
    mu_data: Optional[Union[float, str]] = None,
    peak: Optional[int] = None,
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
    globally_balanced : bool
        Whether the class distribution of the entire dataset is balanced. If False, `mu_data` must be specified.
    locally_balanced : bool
        Whether to use a balanced partition when assigning the dataset to ranks. If False, `mu_partition` must be
        specified.
    shared_test_set : bool
        Whether the test set is private (not shared across subforests). If global_model == False, the test set needs to
        be shared.
    num_samples : int
        The number of samples in the dataset.
    num_classes : int
        The number of classes in the dataset.
    seed_data : int
        The random seed, used for both the dataset generation and the partition and distribution.
    seed_model : int
        The random seed used for the model.
    mu_partition : Optional[Union[float, str]]
        The μ parameter of the skellam distribution for imbalanced class distribution. Has no effect if
        `locally_balanced` is True.
    mu_data : Optional[Union[float, str]]
        The μ parameter of the skellam distribution for imbalanced class distribution in the dataset. Has no effect if
        `globally_balanced` is True.
    peak : Optional[int]
        The position (class index) of the class distribution peak in the dataset. Has no effect if `globally_balanced`
        is True.
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
    assert global_model or shared_test_set

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
            _,
            local_train,
            local_test,
        ) = synthetic_classification_data.generate_and_distribute_synthetic_dataset(
            globally_balanced,
            locally_balanced,
            num_samples,
            num_classes,
            comm.rank,
            comm.size,
            seed_data,
            1 - train_split,
            mu_partition,
            mu_data,
            peak,
            shared_test_set=shared_test_set,
        )
    store_timing(timer)

    print(f"[{comm.rank}/{comm.size}]: Done...")
    print(
        f"Local train samples and targets have shapes {local_train.x.shape} and {local_train.y.shape}."
    )
    print(
        f"Global test samples and targets have shapes {local_test.x.shape} and {local_test.y.shape}."
    )
    print(f"[{comm.rank}/{comm.size}]: Labels are {local_train.y}")

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
        distributed_random_forest.train(local_train.x, local_train.y, global_model)
    store_timing(timer)

    # -------------- Evaluate random forest --------------
    print(f"[{comm.rank}/{comm.size}]: Evaluate random forest.")
    with utils.MPITimer(comm, name="test") as timer:
        distributed_random_forest.test(
            local_test.x, local_test.y, num_classes, global_model
        )
    store_timing(timer)
    store_accuracy(distributed_random_forest, "test")

    if detailed_evaluation:
        distributed_random_forest.test(
            local_train.x, local_train.y, num_classes, global_model
        )
        store_accuracy(distributed_random_forest, "train")

    # -------------- Gather local results, generate dataframe, output collective results --------------
    # convert local results to array, ensure the values are in the same order on all ranks by sorting the keys
    key_order = sorted(local_results.keys())
    local_results_array = np.array([local_results[key] for key in key_order])

    gathered_local_results = comm.gather(local_results_array)
    gathered_class_frequencies_train = local_train.gather_class_frequencies(comm)
    gathered_class_frequencies_test = local_test.gather_class_frequencies(comm)
    if comm.rank == 0:
        # convert arrays back into dicts, then into dataframe
        gathered_local_results = [
            dict(zip(key_order, gathered_values))
            for gathered_values in gathered_local_results
        ]
        results_df = pandas.DataFrame(gathered_local_results + [global_results])

        # add configuration as columns
        for key, value in configuration.items():
            results_df[key] = value

        if output_dir:
            path, base_filename = utils.construct_output_path(
                output_dir, output_label, experiment_id
            )
            utils.save_dataframe(results_df, path / (base_filename + "_results.csv"))
            (
                fig_train,
                _,
            ) = synthetic_classification_data.SyntheticDataset.plot_local_class_distributions(
                gathered_class_frequencies_train
            )
            (
                fig_test,
                _,
            ) = synthetic_classification_data.SyntheticDataset.plot_local_class_distributions(
                gathered_class_frequencies_test
            )
            fig_train.savefig(path / (base_filename + "_class_distribution_train.pdf"))
            fig_test.savefig(path / (base_filename + "_class_distribution_test.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Distributed Random Forests on Synthetic Classification Data"
    )
    parser.add_argument("-rsd", "--random_state_data", type=int, default=0)
    parser.add_argument("-rsf", "--random_state_forest", type=int, default=0)

    parser.add_argument("-t", "--n_trees", type=int, default=100)
    parser.add_argument("-g", "--global_model", action="store_true")

    parser.add_argument("-c", "--n_classes", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=10000)
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
        help="Whether the global dataset has class imbalance. If true, the classes are skellam "
        "distributed. Use --mu_data and --peak to customize the distribution.",
    )
    parser.add_argument(
        "--mu_data",
        type=float,
        default=10,
        help="The μ = μ₁ = μ₂, μ ∊ [0, ∞] parameter of the skellam distribution. The larger μ, the "
        "larger the spread. Edge cases: For μ=0, the peak class has weight 1, while all other "
        "classes have weight 0. For μ=inf, the generated dataset is balanced, i.e., all classes "
        "have equal weights.",
    )
    parser.add_argument(
        "--peak",
        type=int,
        default=0,
        help="The position (class index) of the distribution's peak (i.e. the most frequent class.",
    )

    parser.add_argument(
        "--locally_imbalanced",
        action="store_true",
        help="Whether the partition to local datasets has class imbalance. If true, the classes are "
        "skellam distributed. Use --mu_partition to customize spread of the distributions.",
    )
    parser.add_argument(
        "--mu_partition",
        type=float,
        default=10,
        help="The μ = μ₁ = μ₂, μ ∊ [0, ∞] parameter of the skellam distribution. The larger μ, the "
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
        default=pathlib.Path(__file__).parent.parent / "results",
        help="The directory to write the results to.",
    )
    parser.add_argument(
        "--output_label",
        type=str,
        default="parallel_rf__breaking_iid",
        help="Optional label for the output files.",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        help="Optional subdirectory name to collect related "
        "result in. The subdirectory will be created automatically.",
    )

    config = parser.parse_args()

    train_parallel_on_synthetic_data(
        globally_balanced=not config.globally_imbalanced,
        locally_balanced=not config.locally_imbalanced,
        shared_test_set=not config.private_test_set,
        num_samples=config.num_samples,
        num_classes=config.n_classes,
        seed_data=config.random_state_data,
        seed_model=config.random_state_forest,
        mu_partition=config.mu_partition,
        mu_data=config.mu_data,
        peak=config.peak,
        comm=MPI.COMM_WORLD,
        train_split=config.train_split,
        n_trees=config.n_trees,
        global_model=config.global_model,
        detailed_evaluation=config.detailed_evaluation,
        output_dir=config.output_dir,
        output_label=config.output_label,
        experiment_id=config.experiment_id,
    )
