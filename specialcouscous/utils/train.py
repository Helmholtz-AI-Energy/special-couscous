import logging
import os
import pathlib
import re
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier

from specialcouscous.rf_parallel import DistributedRandomForest
from specialcouscous.synthetic_classification_data import (
    SyntheticDataset,
    generate_and_distribute_synthetic_dataset,
    make_classification_dataset,
)
from specialcouscous.utils import MPITimer, construct_output_path, save_dataframe

log = logging.getLogger(__name__)  # Get logger instance.


def train_serial_on_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_clusters_per_class: int,
    frac_informative: float,
    frac_redundant: float,
    seed_data: int = 0,
    seed_split: int = 0,
    seed_model: int = 0,
    train_split: float = 0.75,
    n_trees: int = 100,
    detailed_evaluation: bool = False,
    output_dir: Optional[Union[str, pathlib.Path]] = None,
    output_label: str = "",
    experiment_id: str = "",
) -> None:
    """
    Train and evaluate a serial random forest on synthetic data.

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
    train_split : float
        Relative size of the train set.
    n_trees : int
        The number of trees in the global forest.
    output_dir : Optional[Union[pathlib.Path, str]]
        Output base directory. If given, the results are written to
        output_dir / year / year-month / date / YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>.
    output_label : str
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment. Default is an empty string.
    """
    configuration = locals()
    for key in ["output_dir"]:
        del configuration[key]
    configuration["comm_size"] = 1

    global_results: Dict[str, Any] = {
        "comm_rank": "global",
        "job_id": int(os.getenv("SLURM_JOB_ID", default=0)),
    }

    log.info("Generating data...")
    # Generate data.
    data_generation_start = time.perf_counter()
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
    data_generation_end = time.perf_counter()
    global_results["time_sec_data_generation"] = (
        data_generation_end - data_generation_start
    )

    log.info(
        f"Done\nTrain samples and targets have shapes {train_samples.shape} and {train_targets.shape}.\n"
        f"First three elements are: {train_samples[:3]} and {train_targets[:3]}\n"
        f"Test samples and targets have shapes {test_samples.shape} and {test_targets.shape}.\n"
        f"First three elements are: {test_samples[:3]} and {test_targets[:3]}\n"
        f"Set up classifier."
    )

    forest_creation_start = time.perf_counter()
    # Set up, train, and test model.
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=seed_model)
    forest_creation_end = time.perf_counter()
    global_results["time_sec_forest_creation"] = (
        forest_creation_end - forest_creation_start
    )

    log.info("Train.")
    train_start = time.perf_counter()
    clf.fit(train_data.x, train_data.y)
    train_end = time.perf_counter()
    global_results["time_sec_training"] = train_end - train_start

    # Calculate accuracies.
    acc_test = clf.score(test_data.x, test_data.y)
    global_results["accuracy_test"] = acc_test

    if detailed_evaluation:
        acc_train = clf.score(train_data.x, train_data.y)
        global_results["accuracy_train"] = acc_train
    log.info(
        f"Time for training is {global_results['time_sec_training']} s.\nTest accuracy is {acc_test}."
    )
    results_df = pandas.DataFrame([global_results])
    # Add configuration as columns.
    for key, value in configuration.items():
        results_df[key] = value

    if output_dir:
        path, base_filename = construct_output_path(
            output_dir, output_label, experiment_id
        )
        save_dataframe(results_df, path / (base_filename + "_results.csv"))
        class_frequencies_train = np.array(
            [
                train_data.get_class_frequency().get(class_id, 0)
                for class_id in range(train_data.n_classes)
            ]
        )
        class_frequencies_test = np.array(
            [
                test_data.get_class_frequency().get(class_id, 0)
                for class_id in range(test_data.n_classes)
            ]
        )
        (
            fig_train,
            _,
        ) = SyntheticDataset.plot_local_class_distributions(
            np.expand_dims(np.array(class_frequencies_train), axis=0)
        )
        (
            fig_test,
            _,
        ) = SyntheticDataset.plot_local_class_distributions(
            np.expand_dims(np.array(class_frequencies_test), axis=0)
        )
        fig_train.savefig(path / (base_filename + "_class_distribution_train.pdf"))
        fig_test.savefig(path / (base_filename + "_class_distribution_test.pdf"))


def train_parallel_on_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    globally_balanced: bool,
    locally_balanced: bool,
    shared_test_set: bool,
    seed_data: int = 0,
    seed_model: int = 0,
    mu_partition: Optional[Union[float, str]] = None,
    mu_data: Optional[Union[float, str]] = None,
    peak: Optional[int] = None,
    make_classification_kwargs: Optional[Dict[str, Any]] = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    train_split: float = 0.75,
    n_trees: int = 100,
    global_model: bool = True,
    detailed_evaluation: bool = False,
    output_dir: Optional[Union[pathlib.Path, str]] = None,
    output_label: str = "",
    experiment_id: str = "",
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
    globally_balanced : bool
        Whether the class distribution of the entire dataset is balanced. If False, `mu_data` must be specified.
    locally_balanced : bool
        Whether to use a balanced partition when assigning the dataset to ranks. If False, `mu_partition` must be
        specified.
    shared_test_set : bool
        Whether the test set is private (not shared across subforests). If global_model == False, the test set needs to
        be shared.
    seed_data : int
        The random seed, used for both the dataset generation and the partition and distribution.
    seed_model : int
        The random seed used for the model.
    mu_partition : Optional[Union[float, str]]
        The μ parameter of the Skellam distribution for imbalanced class distribution. Has no effect if
        ``locally_balanced`` is True.
    mu_data : Optional[Union[float, str]]
        The μ parameter of the Skellam distribution for imbalanced class distribution in the dataset. Has no effect if
        ``globally_balanced`` is True.
    peak : Optional[int]
        The position (class index) of the class distribution peak in the dataset. Has no effect if `globally_balanced`
        is True.
    make_classification_kwargs : Optional[Dict[str, Any]]
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
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
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : Optional[str]
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment. Default is an empty string.
    """
    assert global_model or shared_test_set

    # Get all arguments passed to the function as dict, captures all variables in the current local scope so this needs
    # to be called before defining any other local variables.
    configuration = locals()
    for key in ["comm", "output_dir", "detailed_evaluation"]:
        del configuration[key]
    configuration["comm_size"] = comm.size

    global_results: Dict[str, Any] = {
        "comm_rank": "global",
        "job_id": int(os.getenv("SLURM_JOB_ID", default=0)),
    }
    local_results: Dict[str, Any] = {"comm_rank": comm.rank}

    def store_timing(mpi_timer: MPITimer) -> None:
        """
        Store global and local timing information.

        Parameters
        ----------
        mpi_timer : MPITimer
            A distributed context-manager enabled timer.
        """
        label = "time_sec_" + re.sub(r"\s", "_", mpi_timer.name)
        global_results[label] = mpi_timer.elapsed_time_average
        local_results[label] = mpi_timer.elapsed_time_local

    def store_accuracy(model: DistributedRandomForest, label: str) -> None:
        """
        Store global and local accuracy information.

        Parameters
        ----------
        model : DistributedRandomForest
            A distributed random forest model.
        label : str
            Dataset label, e.g., "train" or "test".
        """
        global_results[f"accuracy_{label}"] = model.acc_global
        local_results[f"accuracy_{label}"] = model.acc_local

    # -------------- Generate and distribute data --------------
    if comm.rank == 0:
        log.info("Generating synthetic data.")
    with MPITimer(comm, name="data generation") as timer:
        (
            _,
            local_train,
            local_test,
        ) = generate_and_distribute_synthetic_dataset(
            globally_balanced,
            locally_balanced,
            n_samples,
            n_features,
            n_classes,
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

    log.info(
        f"[{comm.rank}/{comm.size}]: Done...\n"
        f"Local train samples and targets have shapes {local_train.x.shape} and {local_train.y.shape}.\n"
        f"Global test samples and targets have shapes {local_test.x.shape} and {local_test.y.shape}.\n"
        f"[{comm.rank}/{comm.size}]: Labels are {local_train.y}"
    )

    # -------------- Setup and train random forest --------------
    log.info(f"[{comm.rank}/{comm.size}]: Set up and train local random forest.")
    with MPITimer(comm, name="forest creation") as timer:
        distributed_random_forest = DistributedRandomForest(
            n_trees_global=n_trees,
            comm=comm,
            random_state=seed_model,
            global_model=global_model,
        )
    store_timing(timer)

    with MPITimer(comm, name="training") as timer:
        distributed_random_forest.train(local_train.x, local_train.y, global_model)
    store_timing(timer)

    # -------------- Evaluate random forest --------------
    log.info(f"[{comm.rank}/{comm.size}]: Evaluate random forest.")
    with MPITimer(comm, name="test") as timer:
        distributed_random_forest.test(
            local_test.x, local_test.y, n_classes, global_model
        )
    store_timing(timer)
    store_accuracy(distributed_random_forest, "test")

    if detailed_evaluation:
        distributed_random_forest.test(
            local_train.x, local_train.y, n_classes, global_model
        )
        store_accuracy(distributed_random_forest, "train")

    # -------------- Gather local results, generate dataframe, output collective results --------------
    # Convert local results to array, ensure the values are in the same order on all ranks by sorting the keys.
    key_order = sorted(local_results.keys())
    local_results_array = np.array([local_results[key] for key in key_order])

    gathered_local_results = comm.gather(local_results_array)
    gathered_class_frequencies_train = local_train.gather_class_frequencies(comm)
    gathered_class_frequencies_test = local_test.gather_class_frequencies(comm)
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
            path, base_filename = construct_output_path(
                output_dir, output_label, experiment_id
            )
            save_dataframe(results_df, path / (base_filename + "_results.csv"))
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


def train_parallel_on_balanced_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_clusters_per_class: int,
    frac_informative: float,
    frac_redundant: float,
    seed_data: int = 0,
    seed_split: int = 0,
    seed_model: int = 0,
    mpi_comm: MPI.Comm = MPI.COMM_WORLD,
    train_split: float = 0.75,
    n_trees: int = 100,
    global_model: bool = True,
    detailed_evaluation: bool = False,
    output_dir: Optional[Union[pathlib.Path, str]] = None,
    output_label: str = "",
    experiment_id: str = "",
) -> None:
    """
    Train and evaluate a distributed random forest on globally balanced synthetic data.

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
    mpi_comm : MPI.Comm
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
        'output_dir / year / year-month / date / YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>'.
    output_label : str
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment. Default is an empty string.
    """
    # Get all arguments passed to the function as dict, captures all variables in the current local scope so this needs
    # to be called before defining any other local variables.
    configuration = locals()
    for key in ["mpi_comm", "output_dir", "detailed_evaluation"]:
        del configuration[key]
    configuration["comm_size"] = mpi_comm.size

    global_results: Dict[str, Any] = {
        "comm_rank": "global",
        "job_id": int(os.getenv("SLURM_JOB_ID", default=0)),
    }
    local_results: Dict[str, Any] = {"comm_rank": mpi_comm.rank}

    def store_timing(mpi_timer: MPITimer) -> None:
        """
        Store global and local timing information.

        Parameters
        ----------
        mpi_timer : utils.MPITimer
            A distributed context-manager enabled timer.
        """
        label: str = "time_sec_" + re.sub(r"\s", "_", mpi_timer.name)
        global_results[label] = mpi_timer.elapsed_time_average
        local_results[label] = mpi_timer.elapsed_time_local

    def store_accuracy(model: DistributedRandomForest, label: str) -> None:
        """
        Store global and local accuracy information.

        Parameters
        ----------
        model : DistributedRandomForest
            A distributed random forest model.
        label : str
            Dataset label, e.g., "train" or "test".
        """
        global_results[f"accuracy_{label}"] = model.acc_global
        local_results[f"accuracy_{label}"] = model.acc_local

    # -------------- Generate and distribute data --------------
    if mpi_comm.rank == 0:
        log.info("Generating synthetic data.")
    with MPITimer(mpi_comm, name="data generation") as timer:
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

    log.info(
        f"Done\nTrain samples and targets have shapes {train_data.x.shape} and {train_data.y.shape}.\n"
        f"Test samples and targets have shapes {test_data.x.shape} and {test_data.y.shape}.\n"
        f"Set up classifier."
    )

    # -------------- Setup and train random forest --------------
    log.info(
        f"[{mpi_comm.rank}/{mpi_comm.size}]: Set up and train local random forest."
    )
    with MPITimer(mpi_comm, name="forest creation") as timer:
        distributed_random_forest = DistributedRandomForest(
            n_trees_global=n_trees,
            comm=mpi_comm,
            random_state=seed_model,
            global_model=global_model,
        )
    store_timing(timer)

    with MPITimer(mpi_comm, name="training") as timer:
        if global_model:
            timer_sync_global_model = distributed_random_forest.train(
                train_data.x, train_data.y, global_model
            )
            assert isinstance(timer_sync_global_model, MPITimer)
            store_timing(timer_sync_global_model)
        else:
            distributed_random_forest.train(train_data.x, train_data.y, global_model)
    store_timing(timer)

    # -------------- Evaluate random forest --------------
    log.info(f"[{mpi_comm.rank}/{mpi_comm.size}]: Evaluate random forest.")
    with MPITimer(mpi_comm, name="test") as timer:  # Test trained model on test data.
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

    gathered_local_results = mpi_comm.gather(local_results_array)
    gathered_class_frequencies_train = train_data.gather_class_frequencies(mpi_comm)
    gathered_class_frequencies_test = test_data.gather_class_frequencies(mpi_comm)
    if mpi_comm.rank == 0:
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
            path, base_filename = construct_output_path(
                output_dir, output_label, experiment_id
            )
            save_dataframe(results_df, path / (base_filename + "_results.csv"))
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
