import itertools
import logging
import math
import pathlib
import sys
from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from specialcouscous.synthetic_classification_data import SyntheticDataset
from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")


def compute_and_store_class_frequencies(
    path: pathlib.Path, dataset: SyntheticDataset
) -> np.ndarray:
    """
    Compute the class frequencies of the given dataset, store that as csv at the given path, and return them.

    Parameters
    ----------
    path : pathlib
        The path to store the class frequencies at.
    dataset : SyntheticDataset
        The dataset whose class frequencies shall be analyzed.

    Returns
    -------
    np.ndarray
        The class frequencies.
    """
    class_frequencies_dict = dataset.get_class_frequency()
    class_frequencies = np.array([class_frequencies_dict.get(class_id, 0) for class_id in range(dataset.n_classes)])
    np.savetxt(path, class_frequencies, delimiter=",")
    return class_frequencies


def generate_non_iid_training_sets(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_ranks: int,
    random_state: int | np.random.RandomState,
    test_size: float,
    mu_partition: float | str,
    mu_data: float | str,
    peak: int,
    make_classification_kwargs: dict[str, Any] | None = None,
    stratified_train_test: bool = False,
) -> tuple[SyntheticDataset, dict[int, SyntheticDataset]]:
    """
    Generate the global and local training datasets when using a certain local and global imbalance.

    Parameters
    ----------
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number classes in the dataset.
    n_ranks : int
        The total number of ranks.
    random_state : int | np.random.RandomState
        The random state, used for dataset generation, partition, and distribution.
    test_size : float
        Relative size of the test set.
    mu_partition : float | str, optional
        The μ parameter of the Skellam distribution for imbalanced class distribution.
    mu_data : float | str, optional
        The μ parameter of the Skellam distribution for imbalanced class distribution in the dataset.
    peak : int, optional
        The position (class index) of the class distribution peak in the dataset.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels.

    Returns
    -------
    SyntheticDataset
        The global training dataset.
    dict[int, SyntheticDataset]
        A dict mapping each rank to its local training dataset.
    """
    random_state = check_random_state(random_state)

    log.info("Generating global dataset.")
    log.debug(f"Classification kwargs: {make_classification_kwargs}")
    global_dataset = SyntheticDataset.generate_with_skellam_class_imbalance(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        mu=mu_data,
        peak=peak,
        random_state=random_state,
        make_classification_kwargs=make_classification_kwargs,
    )

    log.info("Generate global train-test split.")
    log.debug(f"Stratify train-test split? {stratified_train_test}")
    global_train_set, _ = global_dataset.train_test_split(
        test_size=test_size,
        stratify=stratified_train_test,
        random_state=random_state,
    )

    log.info("Generating local trainsets.")
    local_train_sets = {}
    for rank in range(n_ranks):
        random_state_partition = np.random.RandomState()
        random_state_partition.set_state(random_state.get_state())
        local_train_sets[rank] = global_train_set.get_local_subset(
            rank,
            n_ranks,
            random_state_partition,
            balanced=False,
            mu=mu_partition,
        )

    return global_train_set, local_train_sets


def store_class_frequencies_train(
    base_path: pathlib.Path,
    n_samples: int,
    n_features: int,
    mu_data: float | str,
    mu_partition: float | str,
    n_nodes: int = 16,
    train_split: float = 0.75,
    random_state_data: int = 0,
    n_classes: int = 10,
) -> None:
    """
    Compute and store the class frequencies of all local and the global training set.

    Parameters
    ----------
    base_path : pathlib.Path
        The base path to write the class frequency csvs to.
    n_samples : int
        The number of samples in the generated dataset.
    n_features : int
        The number of features in the generated dataset.
    mu_data : float | str
        The mu value for imbalanced (global) data generation.
    mu_partition : float | str
        The mu value for imbalanced partition.
    n_nodes : int
        The number of nodes the dataset is distributed over, default 16.
    train_split : float
        The fraction of train samples for the train test split, default 0.75.
    random_state_data : int
        The random state used for data generation, default 0.
    n_classes : int
        The number of classes, default 10.
    """
    dataset = f'n{int(math.log10(n_samples))}_m{int(math.log10(n_features))}'
    dir_name = f"{random_state_data}_{str(mu_data).replace('.', '')}_{str(mu_partition).replace('.', '')}"
    out_dir = base_path / dataset / dir_name
    out_dir.mkdir(exist_ok=True, parents=True)

    log.info(f'Analyzing {dataset=}, {random_state_data=}, {mu_data=}, {mu_partition=}.')
    make_classification_kwargs = {
        "n_clusters_per_class": 1,
        "n_informative": int(0.1 * n_features),
        "n_redundant": int(0.1 * n_features),
        "flip_y": 0.01,
    }
    global_train_class_frequencies = np.zeros(n_classes)

    log.info('Generating and partitioning datasets.')
    global_trainset, local_trainsets = generate_non_iid_training_sets(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_ranks=n_nodes,
        random_state=random_state_data,
        test_size=1 - train_split,
        mu_partition=mu_partition,
        mu_data=mu_data,
        peak=0,
        make_classification_kwargs=make_classification_kwargs,
    )
    log.debug('Done generating dataset.')

    local_freqs = []
    for rank in range(n_nodes):
        path = out_dir / f"class_frequencies_train_rank_{rank}.csv"
        local_freq = compute_and_store_class_frequencies(path, local_trainsets[rank])
        local_freqs.append(local_freq)
        log.debug(f'Local class frequencies written to {path}.')

    path = out_dir / "class_frequencies_train_global.csv"
    compute_and_store_class_frequencies(path, global_trainset)
    log.debug(f'Global class frequencies written to {path}.')

    fig, ax = SyntheticDataset.plot_local_class_distributions(np.array(local_freqs))
    fig.savefig(out_dir / "class_frequencies_train.pdf")

    log.info(f'Done analyzing {dataset=}, {random_state_data=}, {mu_data=}, {mu_partition=}.')
    log.info(f'Results written to {out_dir}.')
    log.info('-' * 80)


if __name__ == "__main__":
    set_logger_config()

    base_path = pathlib.Path(sys.argv[1])
    log.info(f"Writing class frequencies to {base_path}.")
    mus: list[str | float] = ["inf", 2., 0.5]
    datasets = [(1e6, 1e4), (1e7, 1e3)]
    for (n_samples, n_features), mu_data, mu_partition in itertools.product(
        datasets, mus, mus
    ):
        store_class_frequencies_train(
            base_path, int(n_samples), int(n_features), mu_data, mu_partition
        )
