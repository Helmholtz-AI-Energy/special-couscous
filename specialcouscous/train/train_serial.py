import logging
import os
import pathlib
import pickle
import time
from typing import Any

import joblib
import numpy as np
import pandas
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import check_random_state

from specialcouscous.datasets import (
    SyntheticDataset,
    get_dataset,
    make_classification_dataset,
)
from specialcouscous.utils.result_handling import construct_output_path, save_dataframe

log = logging.getLogger(__name__)  # Get logger instance.


def get_confusion_matrix_serial(
    classifier: RandomForestClassifier,
    samples: np.ndarray,
    targets: np.ndarray,
    use_weighted_voting: bool,
    output_path: pathlib.Path,
    base_filename: str,
    label: str,
) -> np.ndarray:
    """
    Calculate and save confusion matrix.

    Parameters
    ----------
    classifier : RandomForestClassifier
        The random forest model.
    samples : np.ndarray
        The samples to evaluate.
    targets : np.ndarray
        The corresponding targets.
    use_weighted_voting : bool
        Whether to use weighted voting as implemented in ``sklearn`` (``True``) or plain voting (``False``).
    output_path : pathlib.Path
        The output directory to save results to.
    base_filename : str
        The base file name, including UUID.
    label : str
        A label, e.g., "train" or "test".

    Returns
    -------
    np.ndarray
        The confusion matrix.
    """
    confusion_matrix_serial = confusion_matrix(
        y_true=targets, y_pred=classifier.predict(samples), normalize=None
    )
    np.savetxt(
        output_path / (base_filename + f"_confusion_matrix_{label}.csv"),
        confusion_matrix_serial,
        delimiter=",",
    )
    return confusion_matrix_serial


def train_serial_on_dataset(
    dataset: str,
    random_state: int | np.random.RandomState = 0,
    random_state_model: int | np.random.RandomState | None = None,
    train_split: float = 0.75,
    stratified_train_test: bool = False,
    data_dir: pathlib.Path | str = pathlib.Path(__file__).parents[2] / "data",
    n_trees: int = 100,
    detailed_evaluation: bool = False,
    output_dir: str | pathlib.Path | None = None,
    output_label: str = "",
    experiment_id: str = "",
    save_model: bool = True,
    use_weighted_voting: bool = False,
) -> None:
    """
    Train and evaluate a serial random forest on the specified dataset.

    Parameters
    ----------
    dataset : str
        The dataset to train and evaluate on. Must be supported by specialcouscous.utils.datasets.
    random_state : int | np.random.RandomState
        The random state used for dataset generation and splitting. If no model-specific random state is provided, it is
        also used to instantiate the random forest classifier.
    random_state_model : int | np.random.RandomState, optional
        An optional random state used for the model.
    train_split : float
        Relative size of the train set.
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels. Default is False.
    data_dir : pathlib.Path | str
        Directory containing the dataset.
    n_trees : int
        The number of trees in the global forest.
    detailed_evaluation : bool
        Whether to perform a detailed evaluation on more than just the local test set.
    output_dir : pathlib.Path | str, optional
        Output base directory. If given, the results are written to
        output_dir / year / year-month / date / YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>.
    output_label : str
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment. Default is an empty string.
    save_model : bool
        Whether the trained classifier is saved to disk (True) or not (False). Default is True.
    use_weighted_voting : bool
        Whether to use weighted voting as implemented in ``sklearn`` (``True``) or plain voting (``False``).
        Default is ``False``.
    """
    configuration = locals()
    for key in ["output_dir", "detailed_evaluation", "data_dir"]:
        del configuration[key]
    configuration["comm_size"] = 1

    global_results: dict[str, Any] = {
        "comm_rank": "global",
        "job_id": int(os.getenv("SLURM_JOB_ID", default=0)),
    }
    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    random_state = check_random_state(random_state)

    assert isinstance(random_state, np.random.RandomState)
    if random_state_model is None:
        random_state_model = random_state.randint(0, np.iinfo(np.int32).max)
        log.info(f"Generated model base seed is {random_state_model}.")
        random_state_model = check_random_state(random_state_model).randint(
            low=0, high=2**32 - 1
        )

    assert output_dir is not None
    output_path, base_filename = construct_output_path(
        output_dir, output_label, experiment_id
    )

    # Generate data.
    log.info("Generating data...")
    data_generation_start = time.perf_counter()
    data = get_dataset(
        dataset, data_dir, random_state, train_split, stratified_train_test
    )
    train_data = SyntheticDataset(x=data.x_train, y=data.y_train)
    test_data = SyntheticDataset(x=data.x_test, y=data.y_test)
    global_results["time_sec_data_generation"] = (
        time.perf_counter() - data_generation_start
    )

    log.info(
        f"Done\nTrain samples and targets have shapes {train_data.x.shape} and {train_data.y.shape}.\n"
        f"First train sample is: \n{train_data.x[0]}\nLast train sample is:\n {train_data.x[-1]}\n"
        f"Test samples and targets have shapes {test_data.x.shape} and {test_data.y.shape}.\n"
        f"First test sample is:\n {test_data.x[0]}\nLast test sample is:\n{test_data.x[-1]}\n"
        f"Set up classifier."
    )

    # Set up, train, and test model.
    forest_creation_start = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=check_random_state(random_state_model),
        n_jobs=-1,
    )
    global_results["time_sec_forest_creation"] = (
        time.perf_counter() - forest_creation_start
    )
    expected_n_jobs = 1 if clf.n_jobs is None else clf.n_jobs
    if expected_n_jobs < 0:
        expected_n_jobs = joblib.cpu_count() + 1 + expected_n_jobs
    log.info(f"Training local random forest with {expected_n_jobs} jobs.")

    log.info("Train.")
    train_start = time.perf_counter()
    clf.fit(train_data.x, train_data.y)
    global_results["time_sec_training"] = time.perf_counter() - train_start

    # Calculate confusion matrix + accuracy.
    global_results["accuracy_test"] = clf.score(test_data.x, test_data.y)

    if data.n_classes == 2:
        prediction_scores = clf.predict_proba(test_data.x)[:, 1]
        global_results["auc_test"] = float(
            sklearn.metrics.roc_auc_score(test_data.y, prediction_scores)
        )
    confusion_matrix_test = get_confusion_matrix_serial(
        classifier=clf,
        samples=test_data.x,
        targets=test_data.y,
        use_weighted_voting=use_weighted_voting,
        output_path=output_path,
        base_filename=base_filename,
        label="test",
    )

    if detailed_evaluation:  # Additionally evaluate on training set.
        global_results["accuracy_train"] = clf.score(train_data.x, train_data.y)
        if data.n_classes == 2:
            prediction_scores = clf.predict_proba(train_data.x)[:, 1]
            global_results["auc_train"] = float(
                sklearn.metrics.roc_auc_score(train_data.y, prediction_scores)
            )
        confusion_matrix_train = get_confusion_matrix_serial(
            classifier=clf,
            samples=train_data.x,
            targets=train_data.y,
            use_weighted_voting=use_weighted_voting,
            output_path=output_path,
            base_filename=base_filename,
            label="train",
        )
    log.info(
        f"Training time is {global_results['time_sec_training']} s.\n"
        f"Test accuracy is {global_results['accuracy_test']}."
        f"Test AUC is {global_results['auc_test']}"
        if data.n_classes == 2
        else ""
    )
    results_df = pandas.DataFrame([global_results])

    for key, value in configuration.items():  # Add configuration as columns.
        results_df[key] = value

    if output_dir:  # Save results to output dir if provided.
        save_results_serial(
            results_df=results_df,
            train_data=train_data,
            test_data=test_data,
            clf=clf,
            output_path=output_path,
            base_filename=base_filename,
            save_model=save_model,
        )


def train_serial_on_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    random_state: int | np.random.RandomState = 0,
    random_state_model: int | np.random.RandomState | None = None,
    make_classification_kwargs: dict[str, Any] | None = None,
    train_split: float = 0.75,
    stratified_train_test: bool = False,
    n_trees: int = 100,
    detailed_evaluation: bool = False,
    output_dir: str | pathlib.Path | None = None,
    output_label: str = "",
    experiment_id: str = "",
    save_model: bool = True,
    use_weighted_voting: bool = False,
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
    random_state : int | np.random.RandomState
        The random state used for dataset generation and splitting. If no model-specific random state is provided, it is
        also used to instantiate the random forest classifier.
    random_state_model : int | np.random.RandomState, optional
        An optional random state used for the model.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
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
        output_dir / year / year-month / date / YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>.
    output_label : str
        Optional label for the csv file, added to the name after the timestamp. Default is an empty string.
    experiment_id : str
        If given, the output file is placed in a further subdirectory <experiment_id> inside the <date> directory.
        Can be used to group the result of multiple runs of an experiment. Default is an empty string.
    save_model : bool
        Whether the trained classifier is saved to disk (True) or not (False). Default is True.
    use_weighted_voting : bool
        Whether to use weighted voting as implemented in ``sklearn`` (``True``) or plain voting (``False``).
        Default is ``False``.
    """
    configuration = locals()
    del configuration["output_dir"]
    configuration["comm_size"] = 1

    global_results: dict[str, Any] = {
        "comm_rank": "global",
        "job_id": int(os.getenv("SLURM_JOB_ID", default=0)),
    }
    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    random_state = check_random_state(random_state)

    assert isinstance(random_state, np.random.RandomState)
    if random_state_model is None:
        random_state_model = random_state.randint(0, np.iinfo(np.int32).max)
        log.info(f"Generated model base seed is {random_state_model}.")
        random_state_model = check_random_state(random_state_model).randint(
            low=0, high=2**32 - 1
        )

    assert output_dir is not None
    output_path, base_filename = construct_output_path(
        output_dir, output_label, experiment_id
    )

    # Generate data.
    log.info("Generating data...")
    data_generation_start = time.perf_counter()
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
    train_data = SyntheticDataset(x=train_samples, y=train_targets)
    test_data = SyntheticDataset(x=test_samples, y=test_targets)
    global_results["time_sec_data_generation"] = (
        time.perf_counter() - data_generation_start
    )

    log.info(
        f"Done\nTrain samples and targets have shapes {train_samples.shape} and {train_targets.shape}.\n"
        f"First train sample is: \n{train_samples[0]}\nLast train sample is:\n {train_samples[-1]}\n"
        f"Test samples and targets have shapes {test_samples.shape} and {test_targets.shape}.\n"
        f"First test sample is:\n {test_samples[0]}\nLast test sample is:\n{test_samples[-1]}\n"
        f"Set up classifier."
    )

    # Set up, train, and test model.
    forest_creation_start = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=check_random_state(random_state_model),
        n_jobs=-1,
    )
    global_results["time_sec_forest_creation"] = (
        time.perf_counter() - forest_creation_start
    )
    expected_n_jobs = 1 if clf.n_jobs is None else clf.n_jobs
    if expected_n_jobs < 0:
        expected_n_jobs = joblib.cpu_count() + 1 + expected_n_jobs
    log.info(f"Training local random forest with {expected_n_jobs} jobs.")

    log.info("Train.")
    train_start = time.perf_counter()
    clf.fit(train_data.x, train_data.y)
    global_results["time_sec_training"] = time.perf_counter() - train_start

    # Calculate confusion matrix + accuracy.
    global_results["accuracy_test"] = clf.score(test_data.x, test_data.y)
    if n_classes == 2:
        prediction_scores = clf.predict_proba(test_data.x)[:, 1]
        global_results["auc_test"] = float(
            sklearn.metrics.roc_auc_score(test_data.y, prediction_scores)
        )
    confusion_matrix_test = get_confusion_matrix_serial(
        classifier=clf,
        samples=test_data.x,
        targets=test_data.y,
        use_weighted_voting=use_weighted_voting,
        output_path=output_path,
        base_filename=base_filename,
        label="test",
    )

    if detailed_evaluation:  # Additionally evaluate on training set.
        global_results["accuracy_train"] = clf.score(train_data.x, train_data.y)
        if n_classes == 2:
            prediction_scores = clf.predict_proba(train_data.x)[:, 1]
            global_results["auc_train"] = float(
                sklearn.metrics.roc_auc_score(train_data.y, prediction_scores)
            )
        confusion_matrix_train = get_confusion_matrix_serial(
            classifier=clf,
            samples=train_data.x,
            targets=train_data.y,
            use_weighted_voting=use_weighted_voting,
            output_path=output_path,
            base_filename=base_filename,
            label="train",
        )
    log.info(
        f"Training time is {global_results['time_sec_training']} s.\n"
        f"Test accuracy is {global_results['accuracy_test']}."
        f"Test AUC is {global_results['auc_test']}"
        if n_classes == 2
        else ""
    )
    results_df = pandas.DataFrame([global_results])

    for key, value in configuration.items():  # Add configuration as columns.
        results_df[key] = value

    if output_dir:  # Save results to output dir if provided.
        save_results_serial(
            results_df=results_df,
            train_data=train_data,
            test_data=test_data,
            clf=clf,
            output_path=output_path,
            base_filename=base_filename,
            save_model=save_model,
        )


def save_results_serial(
    results_df: pandas.DataFrame,
    train_data: SyntheticDataset,
    test_data: SyntheticDataset,
    clf: RandomForestClassifier,
    output_path: pathlib.Path,
    base_filename: str,
    save_model: bool = True,
) -> None:
    """
    Save results of serial random forest training to output directory.

    Parameters
    ----------
    results_df : pandas.DataFrame
        The dataframe containing the results of the experiment.
    train_data : SyntheticDataset
        The synthetic training dataset.
    test_data : SyntheticDataset
        The synthetic test dataset.
    clf : RandomForestClassifier
        The trained random forest classifier.
    output_path : pathlib.Path
        The output directory to save results to.
    base_filename : str
        The base file name, including UUID.
    save_model : bool
        Whether to save the trained random forest classifier model to the output directory.
    """
    save_dataframe(results_df, output_path / (base_filename + "_results.csv"))
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
    fig_train.savefig(output_path / (base_filename + "_class_distribution_train.pdf"))
    fig_test.savefig(output_path / (base_filename + "_class_distribution_test.pdf"))

    if save_model:  # Save model to disk.
        with open(output_path / (base_filename + "_classifier.pickle"), "wb") as f:
            pickle.dump(clf, f, protocol=5)

    plt.close(fig_train)
    plt.close(fig_test)
