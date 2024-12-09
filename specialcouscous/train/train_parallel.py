import logging
import os
import pathlib
import pickle
import re
from typing import Any

import numpy as np
import pandas
from matplotlib import pyplot as plt
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_random_state

from specialcouscous.rf_parallel import DistributedRandomForest
from specialcouscous.synthetic_classification_data import (
    SyntheticDataset,
    generate_and_distribute_synthetic_dataset,
    make_classification_dataset,
)
from specialcouscous.utils.result_handling import construct_output_path, save_dataframe
from specialcouscous.utils.timing import MPITimer

log = logging.getLogger(__name__)  # Get logger instance.


def train_parallel_on_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    globally_balanced: bool,
    locally_balanced: bool,
    shared_test_set: bool,
    random_state: int | np.random.RandomState = 0,
    random_state_model: int | None = None,
    mu_partition: float | str | None = None,
    mu_data: float | str | None = None,
    peak: int | None = None,
    make_classification_kwargs: dict[str, Any] = {},
    comm: MPI.Comm = MPI.COMM_WORLD,
    train_split: float = 0.75,
    stratified_train_test: bool = False,
    n_trees: int = 100,
    shared_global_model: bool = True,
    detailed_evaluation: bool = False,
    output_dir: pathlib.Path | str = "",
    output_label: str = "",
    experiment_id: str = "",
    save_model: bool = True,
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
        Whether the test set is shared across all subforests (True) or private to each rank (False).
        If shared_global_model == False, the test set needs to be shared.
    random_state : int | np.random.RandomState
        The random seed, used for dataset generation, partition, and distribution. Can be  an integer or a numpy random
        state as it must be the same on all ranks to ensure that each rank generates the very same global dataset. If no
        model-specific random state is provided, it is also used to instantiate the random forest classifiers.
    random_state_model : int, optional
        The random seed used for the model. Can only be an integer as it must be different on each rank to ensure that
        each local model is different. In the ``DistributedRandomForest`` constructor, a ``RandomState`` instance seeded
        with this value is used to create a sequence of ``comm.size`` random integers, which are then used to seed a
        different ``RandomState`` instance on each rank passed to the rank-local classifier.
    mu_partition : float | str, optional
        The μ parameter of the Skellam distribution for imbalanced class distribution. Has no effect if
        ``locally_balanced`` is True.
    mu_data : float | str, optional
        The μ parameter of the Skellam distribution for imbalanced class distribution in the dataset. Has no effect if
        ``globally_balanced`` is True.
    peak : int, optional
        The position (class index) of the class distribution peak in the dataset. Has no effect if `globally_balanced`
        is True.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    comm : MPI.Comm
        The MPI communicator to distribute over.
    train_split : float
        Relative size of the train set.
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels.
    n_trees : int
        The number of trees in the global forest.
    shared_global_model : bool
        Whether the local models are all-gathered to one global model shared by all ranks after training.
    detailed_evaluation : bool
        Whether to perform a detailed evaluation on more than just the local test set.
    output_dir : pathlib.Path | str, optional
        Output base directory. If given, the results are written to
        output_dir / year / year-month / date / YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>.
    output_label : str
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str, optional
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment. Default is an empty string.
    save_model : bool
        Whether the locally trained classifiers are saved to disk (True) or not (False). Default is True.
    """
    # Note that to evaluate the global model in a meaningful way, either the model itself of the test data must be
    # shared among all ranks. Otherwise, each rank can only test its local subforest on its private test set, making
    # any evaluation of the global model impossible.
    if not (shared_global_model or shared_test_set):
        raise ValueError(
            "Either `shared_global_model` or `shared_test_set` must be True."
        )

    # Get all arguments passed to the function as dict, captures all variables in the current local scope so this needs
    # to be called before defining any other local variables.
    configuration = locals()
    for key in ["comm", "output_dir", "detailed_evaluation"]:
        del configuration[key]
    configuration["comm_size"] = comm.size

    global_results: dict[str, Any] = {
        "comm_rank": "global",
        "job_id": int(os.getenv("SLURM_JOB_ID", default=0)),
    }
    local_results: dict[str, Any] = {"comm_rank": comm.rank}

    log.debug(f"[{comm.rank}/{comm.size}]: Passed random state is {random_state}.")
    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    random_state = check_random_state(random_state)
    log.debug(
        f"[{comm.rank}/{comm.size}]: The generated random state is:\n{random_state.get_state(legacy=True)}"
    )
    # Generate model base seed if not provided by user.
    assert isinstance(random_state, np.random.RandomState)
    if random_state_model is None:
        random_state_model = random_state.randint(0, np.iinfo(np.int32).max)
        if comm.rank == 0:
            log.info(f"Generated model base seed is {random_state_model}.")

    # -------------- Generate and distribute data --------------
    if comm.rank == 0:
        log.info("Generating synthetic data.")
    with MPITimer(comm, name="data generation") as timer:
        (
            _,
            local_train,
            local_test,
        ) = generate_and_distribute_synthetic_dataset(
            globally_balanced=globally_balanced,
            locally_balanced=locally_balanced,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            rank=comm.rank,
            n_ranks=comm.size,
            random_state=random_state,
            test_size=1 - train_split,
            mu_partition=mu_partition,
            mu_data=mu_data,
            peak=peak,
            make_classification_kwargs=make_classification_kwargs,
            shared_test_set=shared_test_set,
            stratified_train_test=stratified_train_test,
        )
    store_timing(timer, global_results, local_results)
    log.debug(
        f"First train sample is:\n{local_train.x[0]}\nLast train sample is:\n{local_train.x[-1]}\n"
        f""
        f"First test sample is:\n{local_test.x[0]}\nLast test sample is:\n{local_test.x[-1]}"
    )
    log.info(
        f"[{comm.rank}/{comm.size}]: Done...\n"
        f"Local train samples and targets have shapes {local_train.x.shape} and {local_train.y.shape}.\n"
        f"Local test samples and targets have shapes {local_test.x.shape} and {local_test.y.shape}."
    )
    log.debug(f"[{comm.rank}/{comm.size}]: Local test samples are {local_test.x}.")
    log.info("Set up classifier.")

    # -------------- Set up distributed random forest --------------
    log.info(f"[{comm.rank}/{comm.size}]: Set up and train local random forest.")
    with MPITimer(comm, name="forest creation") as timer:
        distributed_random_forest = DistributedRandomForest(
            n_trees_global=n_trees,
            comm=comm,
            random_state=random_state_model,
            shared_global_model=shared_global_model,
        )
    store_timing(timer, global_results, local_results)

    # -------------- Train distributed random forest --------------
    with MPITimer(comm, name="training") as timer:
        distributed_random_forest.train(local_train.x, local_train.y)
    store_timing(timer, global_results, local_results)

    # -------------- Checkpoint trained rank-local subforests --------------
    # Create output directory to save model checkpoints (and configuration + evaluation results later on).
    output_path, base_filename = get_output_path(
        comm, output_dir, output_label, experiment_id
    )
    # Save model to disk.
    if save_model:
        save_model_parallel(
            comm, distributed_random_forest.clf, output_path, base_filename
        )

    # -------------- Gather local results, generate dataframe, output collective results --------------
    # Convert local results to array, ensure the values are in the same order on all ranks by sorting the keys. At this
    # point, only the training times are saved. Note that this dump will be overwritten in the end. However, it serves
    # as a backup in case of errors during evaluation.
    save_results_parallel(
        mpi_comm=comm,
        local_results=local_results,
        global_results=global_results,
        configuration=configuration,
        output_path=output_path,
        base_filename=base_filename,
        test_data=local_test,
        train_data=local_train,
    )
    # -------------- Build shared global model (if applicable) --------------
    if shared_global_model:
        with MPITimer(comm, name="all-gathering model") as timer:
            distributed_random_forest.build_shared_global_model()
        store_timing(timer, global_results, local_results)

    # -------------- Evaluate random forest --------------
    log.info(
        f"[{comm.rank}/{comm.size}]: Evaluate random forest on test dataset with {len(local_test.x)} samples."
    )
    with MPITimer(comm, name="test") as timer:
        distributed_random_forest.evaluate(
            local_test.x, local_test.y, n_classes, shared_global_model
        )
    store_timing(timer, global_results, local_results)
    store_accuracy(distributed_random_forest, "test", global_results, local_results)
    save_confusion_matrix_parallel(
        mpi_comm=comm,
        distributed_forest=distributed_random_forest,
        label="test",
        output_path=output_path,
        base_filename=base_filename,
    )
    # -------------- Gather local results, generate dataframe, output collective results --------------
    # Convert local results to array, ensure the values are in the same order on all ranks by sorting the keys.
    save_results_parallel(
        mpi_comm=comm,
        local_results=local_results,
        global_results=global_results,
        configuration=configuration,
        output_path=output_path,
        base_filename=base_filename,
        test_data=local_test,
        train_data=local_train,
    )

    # -------------- Evaluate trained model also on training data (if applicable) --------------
    if detailed_evaluation:
        log.info(
            f"[{comm.rank}/{comm.size}]: Additionally evaluate on train dataset with {len(local_train.x)} samples."
        )
        if shared_global_model:
            distributed_random_forest.evaluate(
                local_train.x, local_train.y, n_classes, shared_global_model
            )
        else:
            if comm.rank == 0:
                log.info(
                    "The accuracy of the distributed global model cannot "
                    "be calculated without a shared evaluation dataset."
                )
            distributed_random_forest.acc_global = np.nan
            distributed_random_forest.acc_local = distributed_random_forest.clf.score(
                local_train.x, local_train.y
            )
        store_accuracy(
            distributed_random_forest, "train", global_results, local_results
        )
        save_confusion_matrix_parallel(
            mpi_comm=comm,
            distributed_forest=distributed_random_forest,
            label="train",
            output_path=output_path,
            base_filename=base_filename,
        )

        # Save results from detailed evaluation.
        save_results_parallel(
            mpi_comm=comm,
            local_results=local_results,
            global_results=global_results,
            configuration=configuration,
            output_path=output_path,
            base_filename=base_filename,
            test_data=local_test,
            train_data=local_train,
        )


def save_confusion_matrix_parallel(
    mpi_comm: MPI.Comm,
    distributed_forest: DistributedRandomForest,
    label: str,
    output_path: pathlib.Path,
    base_filename: str,
) -> None:
    """
    Save confusion matrices to output directory.

    Parameters
    ----------
    mpi_comm : MPI.Comm
        The MPI communicator to use.
    distributed_forest : DistributedRandomForest
        The distributed random forest classifier.
    output_path : pathlib.Path
        The output directory to save results to.
    base_filename : str
        The base file name, including UUID.
    label : str
        A label, e.g., "train" or "test".
    """
    log.info("Save confusion matrices.")
    # Save each rank's local confusion matrix.
    np.savetxt(
        output_path
        / (base_filename + f"_confusion_matrix_{label}_rank_{mpi_comm.rank}.csv"),
        distributed_forest.confusion_matrix_local,
        delimiter=",",
    )
    if mpi_comm.rank == 0:
        # Save global confusion matrix.
        np.savetxt(
            output_path / (base_filename + f"_confusion_matrix_{label}_global.csv"),
            distributed_forest.confusion_matrix_global,
            delimiter=",",
        )


def save_results_parallel(
    mpi_comm: MPI.Comm,
    local_results: dict[str, Any],
    global_results: dict[str, Any],
    configuration: dict[str, Any],
    output_path: pathlib.Path,
    base_filename: str,
    test_data: SyntheticDataset | None = None,
    train_data: SyntheticDataset | None = None,
) -> None:
    """
    Save results of distributed random forest training to output directory.

    Parameters
    ----------
    mpi_comm : MPI.Comm
        The MPI communicator to use.
    local_results : dict[str, Any]
        Each rank's local results.
    global_results : dict[str, Any]
        The global results.
    configuration : dict[str, Any]
        The experiment configuration.
    output_path : pathlib.Path
        The output directory to save results to.
    base_filename : str
        The base file name, including UUID.
    test_data : SyntheticDataset | None
        The synthetic test dataset.
    train_data : SyntheticDataset | None
        The synthetic training dataset.
    """
    key_order = sorted(local_results.keys())
    local_results_array = np.array([local_results[key] for key in key_order])
    gathered_local_results = mpi_comm.gather(local_results_array)
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

        save_dataframe(results_df, output_path / (base_filename + "_results.csv"))

    if train_data:
        gathered_class_frequencies_train = train_data.allgather_class_frequencies(
            mpi_comm
        )
        if mpi_comm.rank == 0:
            (
                fig_train,
                _,
            ) = SyntheticDataset.plot_local_class_distributions(
                gathered_class_frequencies_train
            )
            fig_train.savefig(
                output_path / (base_filename + "_class_distribution_train.pdf")
            )
            plt.close(fig_train)
    if test_data:
        gathered_class_frequencies_test = test_data.allgather_class_frequencies(
            mpi_comm
        )
        if mpi_comm.rank == 0:
            (
                fig_test,
                _,
            ) = SyntheticDataset.plot_local_class_distributions(
                gathered_class_frequencies_test
            )
            fig_test.savefig(
                output_path / (base_filename + "_class_distribution_test.pdf")
            )
            plt.close(fig_test)


def train_parallel_on_balanced_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    make_classification_kwargs: dict[str, Any] = {},
    random_state: int | np.random.RandomState = 0,
    random_state_model: int | None = None,
    mpi_comm: MPI.Comm = MPI.COMM_WORLD,
    train_split: float = 0.75,
    stratified_train_test: bool = False,
    n_trees: int = 100,
    shared_global_model: bool = True,
    detailed_evaluation: bool = False,
    output_dir: pathlib.Path | str = "",
    output_label: str = "",
    experiment_id: str = "",
    save_model: bool = True,
) -> None:
    """
    Train and evaluate a distributed random forest on globally balanced synthetic data.

    Note that training and test data are not distributed over the ranks but each rank sees the full dataset. Thus, the
    train and test sets on each rank are the same.

    Parameters
    ----------
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number of classes in the dataset.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    random_state : int | np.random.RandomState
        The random seed, used for dataset generation, partition, and distribution. Can be  an integer or a numpy random
        state as it must be the same on all ranks to ensure that each rank generates the very same global dataset. If no
        model-specific random state is provided, it is also used to instantiate the random forest classifiers.
    random_state_model : int, optional
        The random seed used for the model. Can only be an integer as it must be different on each rank to ensure that
        each local model is different. In the ``DistributedRandomForest`` constructor, a ``RandomState`` instance seeded
        with this value is used to create a sequence of ``comm.size`` random integers, which are then used to seed a
        different ``RandomState`` instance on each rank passed to the rank-local classifier.
    mpi_comm : MPI.Comm
        The MPI communicator to distribute over.
    train_split : float
        Relative size of the train set.
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels.
    n_trees : int
        The number of trees in the global forest.
    shared_global_model : bool
        Whether the local models are all-gathered to one global model shared by all ranks after training.
    detailed_evaluation : bool
        Whether to perform a detailed evaluation on more than just the local test set.
    output_dir : pathlib.Path | str, optional
        Output base directory. If given, the results are written to
        'output_dir / year / year-month / date / YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>'.
    output_label : str
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment. Default is an empty string.
    save_model : bool
        Whether the locally trained classifiers are saved to disk (True) or not (False). Default is True.
    """
    # Get all arguments passed to the function as dict, captures all variables in the current local scope so this needs
    # to be called before defining any other local variables.
    configuration = locals()
    for key in ["mpi_comm", "output_dir", "detailed_evaluation"]:
        del configuration[key]
    configuration["comm_size"] = mpi_comm.size

    global_results: dict[str, Any] = {
        "comm_rank": "global",
        "job_id": int(os.getenv("SLURM_JOB_ID", default=0)),
    }
    local_results: dict[str, Any] = {"comm_rank": mpi_comm.rank}

    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    log.debug(
        f"[{mpi_comm.rank}/{mpi_comm.size}]: Passed random state is {random_state}."
    )
    random_state = check_random_state(random_state)
    log.debug(
        f"[{mpi_comm.rank}/{mpi_comm.size}]: The generated random state is:\n{random_state.get_state(legacy=True)}"
    )
    # Generate model base seed if not provided by user.
    assert isinstance(random_state, np.random.RandomState)
    if random_state_model is None:
        random_state_model = random_state.randint(0, np.iinfo(np.int32).max)
        if mpi_comm.rank == 0:
            log.info(f"Generated model base seed is {random_state_model}.")

    # -------------- Generate the data --------------
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
            n_classes=n_classes,
            make_classification_kwargs=make_classification_kwargs,
            random_state=random_state,
            train_split=train_split,
            stratified_train_test=stratified_train_test,
        )
        log.debug(
            f"First train sample is:\n{train_samples[0]}\nLast train sample is:\n{train_samples[-1]}\n"
            f""
            f"First test sample is:\n{test_samples[0]}\nLast test sample is:\n{test_samples[-1]}"
        )
        train_data = SyntheticDataset(x=train_samples, y=train_targets)
        test_data = SyntheticDataset(x=test_samples, y=test_targets)
    store_timing(timer, global_results, local_results)
    log.info(
        f"Done\nTrain samples and targets have shapes {train_data.x.shape} and {train_data.y.shape}.\n"
        f"Test samples and targets have shapes {test_data.x.shape} and {test_data.y.shape}."
    )

    log.debug(
        f"[{mpi_comm.rank}/{mpi_comm.size}]: First two test samples are: \n{test_data.x[0:1]}"
    )

    # -------------- Set up distributed random forest --------------
    log.info(f"[{mpi_comm.rank}/{mpi_comm.size}]: Set up classifier.")
    with MPITimer(mpi_comm, name="forest creation") as timer:
        distributed_random_forest = DistributedRandomForest(
            n_trees_global=n_trees,
            comm=mpi_comm,
            random_state=random_state_model,
            shared_global_model=shared_global_model,
        )
    store_timing(timer, global_results, local_results)

    # -------------- Train distributed random forest --------------
    log.info(f"[{mpi_comm.rank}/{mpi_comm.size}]: Train local random forest.")
    with MPITimer(mpi_comm, name="training") as timer:
        distributed_random_forest.train(train_data.x, train_data.y)
    store_timing(timer, global_results, local_results)

    # -------------- Checkpoint trained rank-local subforests --------------
    # Create output directory to save model checkpoints (and configuration + evaluation results later on).
    output_path, base_filename = get_output_path(
        mpi_comm, output_dir, output_label, experiment_id
    )
    # Save model to disk.
    if save_model:
        save_model_parallel(
            mpi_comm, distributed_random_forest.clf, output_path, base_filename
        )
    # -------------- Gather local results, generate dataframe, output collective results --------------
    # Convert local results to array, ensure the values are in the same order on all ranks by sorting the keys. At this
    # point, only the training times are saved. Note that this dump will be overwritten in the end. However, it serves
    # as a backup in case of errors during evaluation.
    save_results_parallel(
        mpi_comm=mpi_comm,
        local_results=local_results,
        global_results=global_results,
        configuration=configuration,
        output_path=output_path,
        base_filename=base_filename,
        test_data=test_data,
        train_data=train_data,
    )
    # -------------- Build shared global model (if applicable) --------------
    if shared_global_model:
        with MPITimer(mpi_comm, name="all-gathering model") as timer:
            distributed_random_forest.build_shared_global_model()
        store_timing(timer, global_results, local_results)

    # -------------- Evaluate random forest --------------
    log.info(
        f"[{mpi_comm.rank}/{mpi_comm.size}]: Evaluate random forest on test dataset."
    )
    with MPITimer(
        mpi_comm, name="test"
    ) as timer:  # Evaluate trained model on test data.
        distributed_random_forest.evaluate(
            test_data.x, test_data.y, n_classes, shared_global_model
        )
    store_timing(timer, global_results, local_results)
    store_accuracy(distributed_random_forest, "test", global_results, local_results)
    save_confusion_matrix_parallel(
        mpi_comm=mpi_comm,
        distributed_forest=distributed_random_forest,
        label="test",
        output_path=output_path,
        base_filename=base_filename,
    )

    # -------------- Gather local results, generate dataframe, output collective results --------------
    # Convert local results to array, ensure the values are in the same order on all ranks by sorting the keys.
    # Note that in the case of detailed evaluation this dump will be overwritten in the end. However, it serves as a
    # backup in case of errors in the detailed evaluation.
    save_results_parallel(
        mpi_comm=mpi_comm,
        local_results=local_results,
        global_results=global_results,
        configuration=configuration,
        output_path=output_path,
        base_filename=base_filename,
        test_data=test_data,
        train_data=train_data,
    )

    # -------------- Evaluate trained model also on training data (if applicable) --------------
    if detailed_evaluation:
        log.info(
            f"[{mpi_comm.rank}/{mpi_comm.size}]: Additionally evaluate random forest on train dataset."
        )
        distributed_random_forest.evaluate(
            train_data.x, train_data.y, n_classes, shared_global_model
        )
        store_accuracy(
            distributed_random_forest, "train", global_results, local_results
        )
        save_confusion_matrix_parallel(
            mpi_comm=mpi_comm,
            distributed_forest=distributed_random_forest,
            label="train",
            output_path=output_path,
            base_filename=base_filename,
        )
        # Save results from detailed evaluation.
        save_results_parallel(
            mpi_comm=mpi_comm,
            local_results=local_results,
            global_results=global_results,
            configuration=configuration,
            output_path=output_path,
            base_filename=base_filename,
            test_data=test_data,
            train_data=train_data,
        )


def evaluate_parallel_from_checkpoint(
    n_samples: int,
    n_features: int,
    n_classes: int,
    make_classification_kwargs: dict[str, Any] = {},
    random_state: int | np.random.RandomState = 0,
    checkpoint_path: str | pathlib.Path = pathlib.Path("../"),
    checkpoint_uid: str = "",
    random_state_model: int | None = None,
    mpi_comm: MPI.Comm = MPI.COMM_WORLD,
    train_split: float = 0.75,
    stratified_train_test: bool = False,
    n_trees: int = 100,
    detailed_evaluation: bool = False,
    output_dir: pathlib.Path | str = "",
    output_label: str = "",
    experiment_id: str = "",
) -> None:
    """
    Evaluate a distributed random forest loaded from pickled checkpoints on globally balanced synthetic data.

    Note that the data is not distributed over the ranks but each rank sees the full dataset. Thus, the (train and test)
    sets on each rank are the same.

    Parameters
    ----------
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number of classes in the dataset.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    random_state : int | np.random.RandomState
        The random seed, used for dataset generation, partition, and distribution. Can be  an integer or a numpy random
        state as it must be the same on all ranks to ensure that each rank generates the very same global dataset. If no
        model-specific random state is provided, it is also used to instantiate the random forest classifiers.
    checkpoint_path : pathlib.Path | str
        The directory containing the pickled local model checkpoints to load.
    checkpoint_uid : str
        The considered run's unique identifier. Used to identify the correct checkpoints to load.
    random_state_model : int, optional
        The random seed used for the model. Can only be an integer as it must be different on each rank to ensure that
        each local model is different. In the ``DistributedRandomForest`` constructor, a ``RandomState`` instance seeded
        with this value is used to create a sequence of ``comm.size`` random integers, which are then used to seed a
        different ``RandomState`` instance on each rank passed to the rank-local classifier.
    mpi_comm : MPI.Comm
        The MPI communicator to distribute over.
    train_split : float
        Relative size of the train set.
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels. Default is False.
    n_trees : int
        The number of trees in the global forest.
    detailed_evaluation : bool
        Whether to perform a detailed evaluation on more than just the local test set.
    output_dir : pathlib.Path | str, optional
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

    global_results: dict[str, Any] = {
        "comm_rank": "global",
        "job_id": int(os.getenv("SLURM_JOB_ID", default=0)),
    }
    local_results: dict[str, Any] = {"comm_rank": mpi_comm.rank}
    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    log.debug(
        f"[{mpi_comm.rank}/{mpi_comm.size}]: Passed random state is {random_state}."
    )
    random_state = check_random_state(random_state)
    log.debug(
        f"[{mpi_comm.rank}/{mpi_comm.size}]: The generated random state is:\n{random_state.get_state(legacy=True)}"
    )
    # Generate model base seed if not provided by user.
    assert isinstance(random_state, np.random.RandomState)
    if random_state_model is None:
        random_state_model = random_state.randint(0, np.iinfo(np.int32).max)
        if mpi_comm.rank == 0:
            log.info(f"Generated model base seed is {random_state_model}.")

    # -------------- Generate the data --------------
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
            n_classes=n_classes,
            make_classification_kwargs=make_classification_kwargs,
            random_state=random_state,
            train_split=train_split,
            stratified_train_test=stratified_train_test,
        )
        log.debug(
            f"First train sample is:\n{train_samples[0]}\nLast train sample is:\n{train_samples[-1]}\n"
            f""
            f"First test sample is:\n{test_samples[0]}\nLast test sample is:\n{test_samples[-1]}"
        )
        log.info(
            f"Done\nTrain samples and targets have shapes {train_samples.shape} and {train_targets.shape}.\n"
            f"Test samples and targets have shapes {test_samples.shape} and {test_targets.shape}."
        )
        if detailed_evaluation:  # Only keep training data for detailed evalution.
            train_data = SyntheticDataset(x=train_samples, y=train_targets)
        else:  # Delete otherwise.
            log.info(f"[{mpi_comm.rank}/{mpi_comm.size}]: Delete training data.")
            del train_samples, train_targets
            train_data = None
        test_data = SyntheticDataset(x=test_samples, y=test_targets)
        log.debug(
            f"[{mpi_comm.rank}/{mpi_comm.size}]: First two test samples are: \n{test_data.x[0:1]}"
        )
    store_timing(timer, global_results, local_results)

    # -------------- Set up distributed random forest --------------
    log.info(f"[{mpi_comm.rank}/{mpi_comm.size}]: Set up classifier.")
    with MPITimer(mpi_comm, name="forest creation") as timer:
        distributed_random_forest = DistributedRandomForest(
            n_trees_global=n_trees,
            comm=mpi_comm,
            shared_global_model=False,
        )
    store_timing(timer, global_results, local_results)

    # Load pickled model checkpoints.
    distributed_random_forest.load_checkpoints(checkpoint_path, checkpoint_uid)

    # Create output directory to save model checkpoints (and configuration + evaluation results later on).
    output_path, base_filename = get_output_path(
        mpi_comm, output_dir, output_label, experiment_id
    )

    # -------------- Gather local results, generate dataframe, output collective results --------------
    # Convert local results to array, ensure the values are in the same order on all ranks by sorting the keys. At this
    # point, only the training times are saved. Note that this dump will be overwritten in the end. However, it serves
    # as a backup in case of errors during evaluation.
    save_results_parallel(
        mpi_comm=mpi_comm,
        local_results=local_results,
        global_results=global_results,
        configuration=configuration,
        output_path=output_path,
        base_filename=base_filename,
        test_data=test_data,
        train_data=train_data,
    )

    # -------------- Evaluate random forest --------------
    log.info(
        f"[{mpi_comm.rank}/{mpi_comm.size}]: Evaluate random forest on test dataset."
    )
    with MPITimer(
        mpi_comm, name="test"
    ) as timer:  # Evaluate trained model on test data.
        distributed_random_forest.evaluate(
            samples=test_data.x,
            targets=test_data.y,
            n_classes=n_classes,
            shared_global_model=False,
        )
    store_timing(timer, global_results, local_results)
    store_accuracy(distributed_random_forest, "test", global_results, local_results)
    save_confusion_matrix_parallel(
        mpi_comm=mpi_comm,
        distributed_forest=distributed_random_forest,
        label="test",
        output_path=output_path,
        base_filename=base_filename,
    )
    # -------------- Gather local results, generate dataframe, output collective results --------------
    # Convert local results to array, ensure the values are in the same order on all ranks by sorting the keys.
    # Note that in the case of detailed evaluation this dump will be overwritten in the end. However, it serves as a
    # backup in case of errors in the detailed evaluation.
    save_results_parallel(
        mpi_comm=mpi_comm,
        local_results=local_results,
        global_results=global_results,
        configuration=configuration,
        output_path=output_path,
        base_filename=base_filename,
        test_data=test_data,
        train_data=train_data,
    )

    # -------------- Evaluate trained model also on training data (if applicable) --------------
    if detailed_evaluation:
        log.info(
            f"[{mpi_comm.rank}/{mpi_comm.size}]: Additionally evaluate random forest on train dataset."
        )
        distributed_random_forest.evaluate(
            samples=train_data.x,  # type:ignore
            targets=train_data.y,  # type:ignore
            n_classes=n_classes,
            shared_global_model=False,
        )
        store_accuracy(
            distributed_random_forest, "train", global_results, local_results
        )
        save_confusion_matrix_parallel(
            mpi_comm=mpi_comm,
            distributed_forest=distributed_random_forest,
            label="train",
            output_path=output_path,
            base_filename=base_filename,
        )
        # Save results from detailed evaluation.
        save_results_parallel(
            mpi_comm=mpi_comm,
            local_results=local_results,
            global_results=global_results,
            configuration=configuration,
            output_path=output_path,
            base_filename=base_filename,
            test_data=test_data,
            train_data=train_data,
        )


def save_model_parallel(
    mpi_comm: MPI.Comm,
    clf: RandomForestClassifier,
    path: pathlib.Path,
    base_filename: str,
) -> None:
    """
    Save rank-local random forest classifiers to output directory.

    Parameters
    ----------
    mpi_comm : MPI.Comm
        The MPI communicator to use.
    clf : RandomForestClassifier
        The local trained random forest classifier.
    path : pathlib.Path
        The output directory to save results to.
    base_filename : str
        The base file name, including UUID.
    """
    with open(
        path / (base_filename + f"_classifier_rank_{mpi_comm.rank}.pickle"),
        "wb",
    ) as f:
        pickle.dump(clf, f, protocol=5)


def store_timing(
    mpi_timer: MPITimer, global_results: dict[str, Any], local_results: dict[str, Any]
) -> None:
    """
    Store global and local timing information.

    Parameters
    ----------
    mpi_timer : MPITimer
        A distributed context-manager enabled timer.
    global_results : dict[str, Any]
        The global results dictionary.
    local_results : dict[str, Any]
        The local results dictionary.
    """
    label = "time_sec_" + re.sub(r"\s", "_", mpi_timer.name)
    global_results[label] = mpi_timer.elapsed_time_average
    local_results[label] = mpi_timer.elapsed_time_local


def store_accuracy(
    model: DistributedRandomForest,
    label: str,
    global_results: dict[str, Any],
    local_results: dict[str, Any],
) -> None:
    """
    Store global and local accuracy information.

    Parameters
    ----------
    model : DistributedRandomForest
        A distributed random forest model.
    label : str
        Dataset label, e.g., "train" or "test".
    global_results : dict[str, Any]
        The global results dictionary.
    local_results : dict[str, Any]
        The local results dictionary.
    """
    global_results[f"accuracy_global_{label}"] = model.acc_global
    local_results[f"accuracy_global_local_{label}"] = model.acc_global_local
    local_results[f"accuracy_local_{label}"] = model.acc_local


def get_output_path(
    comm: MPI.Comm,
    output_dir: str | pathlib.Path,
    output_label: str = "",
    experiment_id: str = "",
) -> tuple[pathlib.Path, str]:
    """
    Create output directory to save model checkpoints (and configuration + evaluation results later on).

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator.
    output_dir : str | pathlib.Path
        The root output directory.
    output_label : str, optional
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str, optional
        If this is given, the file is placed in a further subdirectory of that name, i.e.,
        'output_path / year / year-month / date / experiment_id / <filename>.csv'. This can be used to group multiple
        runs of an experiment. Default is an empty string.

    Returns
    -------
    pathlib.Path
        The full output directory path.
    str
        The global base file name for this run.
    """
    # Create globally unique full output path and base file name only on root.
    # Otherwise, each rank would create its own UUID.
    if comm.rank == 0:
        path, base_filename = construct_output_path(
            output_dir, output_label, experiment_id
        )
    else:
        path, base_filename = pathlib.Path(""), ""
    # Broadcast to all ranks and return the same path and base file name on all ranks.
    return pathlib.Path(comm.bcast(path, root=0)), comm.bcast(base_filename, root=0)
