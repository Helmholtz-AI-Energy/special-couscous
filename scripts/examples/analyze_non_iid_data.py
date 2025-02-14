import itertools
import logging
import math
import pathlib
import sys

import numpy as np
from sklearn.utils import check_random_state

from specialcouscous.synthetic_classification_data import (
    SyntheticDataset,
    generate_and_distribute_synthetic_dataset,
)
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
    class_frequencies = np.zeros(dataset.n_classes)
    for cls, frequency in class_frequencies_dict.items():
        class_frequencies[int(cls)] = frequency

    np.savetxt(path, class_frequencies, delimiter=",")
    return class_frequencies


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
    dataset = f'n{math.log10(n_samples)}_m{math.log10(n_features)}'
    dir_name = f"{random_state_data}_{str(mu_data).replace('.', '')}_{str(mu_partition).replace('.', '')}"
    out_dir = base_path / dataset / dir_name
    out_dir.mkdir(exist_ok=True, parents=True)
    log.info(f'Analyzing {dataset=}, {random_state_data=}, {mu_data=}, {mu_partition=}.')

    random_state_data = check_random_state(random_state_data)
    make_classification_kwargs = {
        "n_clusters_per_class": 1,
        "n_informative": int(0.1 * n_features),
        "n_redundant": int(0.1 * n_features),
        "flip_y": 0.01,
    }
    global_train_class_frequencies = np.zeros(n_classes)

    for rank in range(n_nodes):
        log.debug(f'Generating dataset for {rank=}.')
        (_, local_train, local_test) = generate_and_distribute_synthetic_dataset(
            globally_balanced=False,
            locally_balanced=False,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            rank=rank,
            n_ranks=n_nodes,
            random_state=random_state_data,
            test_size=1 - train_split,
            mu_partition=mu_partition,
            mu_data=mu_data,
            peak=0,
            make_classification_kwargs=make_classification_kwargs,
        )
        log.debug(f'Done generating dataset.')

        path = out_dir / f"class_frequencies_train_rank_{rank}.csv"
        global_train_class_frequencies += compute_and_store_class_frequencies(
            path, local_train
        )
        log.info(f'Local class frequencies written to {path}.')

    path = out_dir / "class_frequencies_train_global.csv"
    np.savetxt(path, global_train_class_frequencies, delimiter=",")
    log.info(f'Global class frequencies written to {path}.')


if __name__ == "__main__":
    set_logger_config(level=logging.DEBUG)

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
