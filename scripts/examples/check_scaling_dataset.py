import argparse
import logging
import pathlib
from typing import cast

from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier

from specialcouscous.rf_parallel import DistributedRandomForest
from specialcouscous.scaling_dataset import read_scaling_dataset_from_hdf5
from specialcouscous.synthetic_classification_data import SyntheticDataset
from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")


def train_and_evaluate_on_dataset(
    train_data: SyntheticDataset,
    n_trees: int,
    comm: MPI.Comm | None = None,
    test_data: SyntheticDataset | None = None,
    global_model: bool = True,
) -> None:
    """
    Train and evaluate a random forest on the given datasets.

    If an MPI communicator is given, a distributed random forest is trained, otherwise a serial RandomForestClassifier
    is trained.

    Parameters
    ----------
    train_data : SyntheticDataset
        The dataset to train on.
    n_trees : int
        The number of trees to train.
    comm : MPI.Comm | None
        Optional MPI communicator, if given, a distributed random forest is trained and evaluated.
    test_data : SyntheticDataset | None
        Optional test data to evaluate the trained forest on.
    global_model : bool
        Whether to aggregate a shared global model for evaluation when training a distributed forest. If False,
        the global model cannot be evaluated on local test data.
    """
    if comm is None:
        log.info(f"Training serial random forest with {n_trees}")
        random_forest = RandomForestClassifier(n_estimators=n_trees, random_state=0)
        random_forest.fit(train_data.x, train_data.y)

        train_accuracy = random_forest.score(train_data.x, train_data.y)

        if test_data is not None:
            test_accuracy = random_forest.score(test_data.x, test_data.y)
    else:
        log.info(f"Training parallel random forest with {n_trees} on {comm.size} ranks")
        random_forest = DistributedRandomForest(
            n_trees_global=n_trees,
            comm=comm,
            random_state=0,
            shared_global_model=global_model,
        )
        random_forest.train(train_data.x, train_data.y)
        if global_model:
            random_forest.build_shared_global_model()

        random_forest.evaluate(
            train_data.x,
            train_data.y,
            train_data.n_classes,
            shared_global_model=global_model,
        )
        train_accuracy = random_forest.acc_global

        if test_data is not None:
            random_forest.evaluate(
                test_data.x,
                test_data.y,
                test_data.n_classes,
                shared_global_model=global_model,
            )
            test_accuracy = random_forest.acc_global

    log.info(f"Train accuracy is {train_accuracy:.2%}")
    if test_data is not None:
        log.info(f"Test accuracy is {test_accuracy:.2%}")


def load_and_check_dataset(
    dataset: str,
    data_root_path: str,
    n_ranks: int = 64,
    seed: int = 0,
    n_trees: int | None = None,
    comm: MPI.Comm | None = None,
) -> None:
    """
    Load and check a scaling dataset by loading the HDF5 file rank by rank and printing the dataset sizes.

    Parameters
    ----------
    dataset : str
        Name of the baseline single-node dataset (n6m4 or n7m3).
    data_root_path : str
        The directory containing the datasets.
    n_ranks : int
        The maximum number of ranks, the check will load slices range(n_ranks) from the HDF5. Default is 64.
    seed : int
        Seed of the dataset to load, default is 0.
    n_trees : int | None
        Optional number of trees. If given, a forest containing this many trees is trained on evaluated on the loaded
        data to verify the data can be learned.
    comm : MPI.Comm | None
        Optional MPI communicator for forest training. If this and n_trees are given, a distributed random forest is
        trained and evaluated on the loaded dataset.
    """
    dataset_dirnames = {
        "n6m4": "n_samples_48250000__n_features_10000__n_classes_10",
        "n7m3": "n_samples_482500000__n_features_1000__n_classes_10",
    }

    dirname = dataset_dirnames.get(dataset, dataset)
    hdf5_path = (
        pathlib.Path(data_root_path) / dirname / f"{n_ranks}_ranks__seed_{seed}.h5"
    )
    log.info(f"Loading dataset from {hdf5_path}")

    for rank in range(n_ranks):
        local_train_set, global_test_set, root_attrs = read_scaling_dataset_from_hdf5(
            pathlib.Path(hdf5_path),
            rank=rank,
            with_global_test=(rank == 0) or (n_trees is not None),
        )
        local_train_set = cast(SyntheticDataset, local_train_set)

        if rank == 0:
            log.info(f"Root attrs:\n{root_attrs}\n")
            global_test_set = cast(SyntheticDataset, global_test_set)
            log.info(
                f"Global test set: {global_test_set.n_samples} samples "
                f"with {global_test_set.x.shape[1]} features "
                f"and {global_test_set.n_classes} classes.\n"
            )

        log.info(
            f"Rank {rank:2d}: Local train set: {local_train_set.n_samples} samples."
        )

        if n_trees is not None:
            log.info(
                f"Rank {rank:2d}: local training and evaluation on global test set."
            )
            train_and_evaluate_on_dataset(
                local_train_set, n_trees, None, global_test_set
            )

    if comm is not None and n_trees is not None:
        log.info(
            f"Parallel training and evaluation on global test set with {comm.size} ranks."
        )
        assert comm.size <= n_ranks
        local_train_set, global_test_set, root_attrs = read_scaling_dataset_from_hdf5(
            pathlib.Path(hdf5_path), rank=comm.rank
        )
        local_train_set = cast(SyntheticDataset, local_train_set)
        train_and_evaluate_on_dataset(local_train_set, n_trees, comm, global_test_set)


if __name__ == "__main__":
    set_logger_config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Select the scaling dataset to check based on the name of it's single-node baseline",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/hkfs/work/workspace/scratch/ku4408-SpecialCouscous/datasets/",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n_ranks", type=int, default=64)
    parser.add_argument("--n_trees", type=int)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD if args.parallel else None
    load_and_check_dataset(
        args.dataset, args.datapath, args.n_ranks, n_trees=args.n_trees, comm=comm
    )
