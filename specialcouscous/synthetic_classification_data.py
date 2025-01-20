from __future__ import annotations
import collections
import itertools
import logging
import os
import pathlib
from typing import Any, Callable, Type

import h5py
import numpy as np
import numpy.typing as npt
import scipy
from matplotlib import pyplot as plt
from mpi4py import MPI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from specialcouscous.utils.plot import (
    plot_class_distributions,
    plot_local_class_distributions_as_ridgeplot,
)

log = logging.getLogger(__name__)  # Get logger instance.


class DatasetPartition:
    """
    Dataset partitioner.

    Attributes
    ----------
    indices_by_class : dict[int, np.ndarray]
        The sample indices grouped by classes. Key is class and value is an array containing the indices of samples
        belonging to that class.
    n_classes : int
        The number of classes in the dataset.
    n_samples : int
        The number of samples in the dataset.
    targets : numpy.ndarray
        A 1D numpy array containing the target for each sample.

    Methods
    -------
    get_class_imbalance()
        Get the percentage share for each target value from an array of targets.
    assigned_indices_by_rank()
        Convert array assigning each sample index to a rank to dict mapping each rank to array of its assigned indices.
    weighted_partition()
        Partition a number of elements into len(weights) parts according to the weights (as closely as possible).
    deterministic_class_partitioner()
        Return a function which partitions the samples for a class based on optional class weights.
    sampling_class_partitioner()
        Return a function which partitions the samples for a class based on optional class weights.
    partition()
        Partition this dataset among ``n_rank`` ranks according to the given class weights.
    balanced_partition()
        Create balanced partition of this dataset among ``n_rank`` ranks.
    shifted_skellam_imbalanced_partition()
        Partition this dataset among ``n_rank`` ranks using an imbalanced shifted Skellam class distribution.
    """

    def __init__(self, targets: np.ndarray) -> None:
        """
        Initialize the partition with the targets for each sample.

        A list of sample indices for each class is determined automatically.

        Parameters
        ----------
        targets : numpy.ndarray
            A 1D numpy array containing the target for each sample.
        """
        self.targets = targets
        self.n_samples = len(self.targets)
        self.indices_by_class = {
            class_index: np.nonzero(self.targets == class_index)[0]
            for class_index in np.unique(self.targets)
        }
        self.n_classes = len(self.indices_by_class.keys())

    def get_class_imbalance(self) -> dict[int, float]:
        """
        Get the percentage share for each target value from an array of targets.

        Returns
        -------
        dict[int, float]
            A dict mapping each target value to its percentage.
        """
        unique, counts = np.unique(self.targets, return_counts=True)
        counts = counts / counts.sum()
        return dict(zip(unique, counts))

    @staticmethod
    def assigned_indices_by_rank(
        assigned_ranks: npt.NDArray[np.int32],
    ) -> dict[int, npt.NDArray[np.int32]]:
        """
        Convert array assigning each sample index to a rank to dict mapping each rank to array of its assigned indices.

        Parameters
        ----------
        assigned_ranks : numpy.ndarray[int]
            Numpy array of length ``n_samples``, assigning each sample to a rank.

        Returns
        -------
        dict[int, numpy.ndarray[int]]
            A dict mapping each rank to the assigned sample indices.
        """
        return {
            rank: np.nonzero(assigned_ranks == rank)[0]
            for rank in np.unique(assigned_ranks)
        }

    @staticmethod
    def weighted_partition(
        total_elements: int, weights: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.int32]:
        """
        Partition a number of elements into ``len(weights)`` parts according to the ``weights`` as closely as possible.

        Parameters
        ----------
        total_elements : int
            The number of elements to partition.
        weights : numpy.ndarray[float]
            The weight for each part.

        Returns
        -------
        numpy.ndarray[int]
            The number of elements in each part.
        """
        percentage = weights / weights.sum()
        element_count_per_part = np.floor(percentage * total_elements).astype(int)

        # Assign remaining elements to the parts with the largest difference between actual and current weight.
        remainders = percentage * total_elements - element_count_per_part
        remaining_elements = total_elements - element_count_per_part.sum()
        if remaining_elements > 0:
            index_in_increasing_weight = np.argsort(remainders)
            element_count_per_part[
                index_in_increasing_weight[-remaining_elements:]
            ] += 1

        return element_count_per_part

    def _partition_class_wise(
        self,
        class_partitioner: Callable[
            [int, npt.NDArray[np.int32]], npt.NDArray[np.int32]
        ],
    ) -> npt.NDArray[np.int32]:
        """
        Partition the dataset one class at a time using the given class_partitioner.

        Parameters
        ----------
        class_partitioner : Callable[[int, numpy.ndarray[int]], ndarray[int]]
            A function mapping a class index and an array containing the indices of samples belonging to that class to
            the assigned rank for each sample.

        Returns
        -------
        numpy.ndarray[int]
            The rank assigned to each sample.
        """
        assigned_ranks_by_class = {
            c: class_partitioner(c, samples)
            for c, samples in self.indices_by_class.items()
        }

        # Not the most performant but it's only called once.
        index_rank_pairs = [
            (sample_index, rank)
            for c in assigned_ranks_by_class.keys()
            for (sample_index, rank) in zip(
                self.indices_by_class[c], assigned_ranks_by_class[c]
            )
        ]
        assigned_ranks = np.repeat(-1, self.n_samples)
        for sample_index, rank in index_rank_pairs:
            assigned_ranks[sample_index] = rank

        if np.any(assigned_ranks == -1):
            raise RuntimeError(
                "Expected all samples to be assigned to a rank but samples "
                f"{np.nonzero(assigned_ranks == -1)[0]} have default rank -1."
            )
        return assigned_ranks

    @staticmethod
    def _class_weights_to_rank_weights(
        class_weights_by_rank: dict[int, npt.NDArray[np.float32]],
    ) -> dict[int, npt.NDArray[np.float32]]:
        """
        Convert a dict containing the class balance for each rank to a dict containing the rank balance for each class.

        Parameters
        ----------
        class_weights_by_rank : dict[int, numpy.ndarray[float]]
            The class balance for each rank as dict rank -> class weights.

        Returns
        -------
        dict[int, numpy.ndarray[float]]
            The rank balance (in percent) for each class as dict class -> rank_weights.
        """
        n_ranks = len(class_weights_by_rank)
        n_classes = len(class_weights_by_rank[0])
        rank_weights_by_class = {
            c: np.array([class_weights_by_rank[rank][c] for rank in range(n_ranks)])
            for c in range(n_classes)
        }
        # Convert to probabilities and return.
        return {
            c: rank_weights / rank_weights.sum()
            for c, rank_weights in rank_weights_by_class.items()
        }

    @staticmethod
    def deterministic_class_partitioner(
        n_ranks: int,
        class_weights_by_rank: dict[int, npt.NDArray[np.float32]] | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> Callable[[int, npt.NDArray[np.int32]], npt.NDArray[np.int32]]:
        """
        Return a function which partitions the samples for a class based on optional class weights.

        The partitioning is done by calculating the number of samples per rank and shuffling the assignment.

        Parameters
        ----------
        n_ranks : int
            The number of ranks to partition into.
        class_weights_by_rank : dict[int, numpy.ndarray[float]], optional
            The class balance for each rank as dict rank -> class_weights. If None, a uniform class balance is assumed.
        random_state : int | numpy.random.RandomState, optional
            The random state used for the shuffling.

        Returns
        -------
        Callable[[int, numpy.ndarray[int]], numpy.ndarray[int]]
            A functon mapping the class index and sample indices to the assigned rank by sample.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state_rng = check_random_state(random_state)
        rank_probabilities_by_class: dict
        if class_weights_by_rank is None:
            rank_probabilities_by_class = collections.defaultdict(
                lambda: np.repeat(1 / n_ranks, n_ranks)
            )
        else:
            rank_probabilities_by_class = (
                DatasetPartition._class_weights_to_rank_weights(class_weights_by_rank)
            )

        def assign_to_ranks(
            class_index: int, sample_indices: npt.NDArray[np.int32]
        ) -> npt.NDArray[np.int32]:
            elements_per_rank = DatasetPartition.weighted_partition(
                len(sample_indices), rank_probabilities_by_class[class_index]
            )
            assigned_ranks = np.array(
                [
                    rank
                    for rank, n_elements in enumerate(elements_per_rank)
                    for _ in range(n_elements)
                ]
            )
            random_state_rng.shuffle(assigned_ranks)
            return assigned_ranks

        return assign_to_ranks

    @staticmethod
    def sampling_class_partitioner(
        n_ranks: int,
        class_weights_by_rank: dict[int, npt.NDArray[np.float32]] | None = None,
        random_state: int | np.random.RandomState | None = None,
        **choice_kwargs: Any,
    ) -> Callable[[int, npt.NDArray[np.int32]], npt.NDArray[np.int32]]:
        """
        Return a function which partitions the samples for a class based on optional class weights.

        The partitioning is done by sampling a random rank based on its probability.

        Parameters
        ----------
        n_ranks : int
            The number of ranks to partition into.
        class_weights_by_rank : dict[int, numpy.ndarray[float]], optional
            The class balance for each rank as dict rank -> class_weights. If None, a uniform class balance is assumed.
        random_state : int | np.random.RandomState, optional
            The random state used for the random sampling.
        **choice_kwargs : Any
            Optional additional keyword arguments to ``numpy``'s ``rng.choice``.

        Returns
        -------
        Callable[[int, numpy.ndarray[int]], ndarray[int]]
            A function mapping the class index and sample indices to the assigned rank by sample.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state_rng = check_random_state(random_state)

        rank_probabilities_by_class: dict
        if class_weights_by_rank is None:
            rank_probabilities_by_class = collections.defaultdict(None)
        else:
            rank_probabilities_by_class = (
                DatasetPartition._class_weights_to_rank_weights(class_weights_by_rank)
            )

        def assign_to_ranks(
            class_index: int, sample_indices: npt.NDArray[np.int32]
        ) -> npt.NDArray[np.int32]:
            return random_state_rng.choice(
                n_ranks,
                len(sample_indices),
                p=rank_probabilities_by_class[class_index],
                **choice_kwargs,
            )

        return assign_to_ranks

    def partition(
        self,
        n_ranks: int,
        class_weights_by_rank: dict[int, npt.NDArray[np.float32]] | None = None,
        random_state: int | np.random.RandomState | None = None,
        sampling: bool = False,
    ) -> npt.NDArray[np.int32]:
        """
        Partition this dataset among ``n_rank`` ranks according to the given class weights.

        Parameters
        ----------
        n_ranks : int
            The number of ranks to partition into.
        class_weights_by_rank : dict[int, numpy.ndarray[float]] | None
            The class balance for each rank as dict rank -> class_weights. If None, a uniform class balance is assumed.
        random_state : int | np.random.RandomState
            The random state used for shuffling and random sampling.
        sampling : bool
            Whether to partition the dataset using deterministic element counts and shuffling (number of elements per
            rank and class independent of random seed, only which samples are assigned to which rank may change) or
            random sampling (number of elements per rank and class is expected to be the corresponding weight but may
            change slightly with each random seed).

        Returns
        -------
        numpy.ndarray[int]
            The rank assigned to each sample.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state = check_random_state(random_state)
        if sampling:
            class_partitioner = self.sampling_class_partitioner(
                n_ranks, class_weights_by_rank, random_state
            )
        else:
            class_partitioner = self.deterministic_class_partitioner(
                n_ranks, class_weights_by_rank, random_state
            )
        return self._partition_class_wise(class_partitioner)

    def balanced_partition(
        self,
        n_ranks: int,
        random_state: int | np.random.RandomState | None = None,
        sampling: bool = False,
    ) -> npt.NDArray[np.int32]:
        """
        Create balanced partition of this dataset among ``n_rank`` ranks.

        Parameters
        ----------
        n_ranks : int
            The number of ranks to partition into.
        random_state : int | np.random.RandomState, optional
            The random seed used for shuffling and random sampling.
        sampling : bool
            Whether to partition the dataset using deterministic element counts and shuffling (number of elements per
            rank and class independent of random seed, only which samples are assigned to which rank may change) or
            random sampling (number of elements per rank and class is expected to be the corresponding weight but may
            change slightly with each random seed).

        Returns
        -------
        numpy.ndarray[int]
            The rank assigned to each sample.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state = check_random_state(random_state)
        return self.partition(n_ranks, None, random_state, sampling)

    def shifted_skellam_imbalanced_partition(
        self,
        n_ranks: int,
        mu: float | str,
        random_state: int | np.random.RandomState | None = None,
        sampling: bool = False,
    ) -> npt.NDArray[np.int32]:
        """
        Partition this dataset among ``n_rank`` ranks using an imbalanced class distribution.

        A shifted Skellam distribution is used to model the imbalances.

        Parameters
        ----------
        n_ranks : int
            The number of ranks to partition into.
        mu : float | str
            The μ = μ₁ = μ₂ parameter of the Skellam distribution. Must be ≥0 or 'inf'.
        random_state : int | np.random.RandomState, optional
            The random seed used for shuffling and random sampling.
        sampling : bool
            Whether to partition the dataset using deterministic element counts and shuffling (number of elements per
            rank and class independent of random seed, only which samples are assigned to which rank may change) or
            random sampling (number of elements per rank and class is expected to be the corresponding weight but may
            change slightly with each random seed).

        Returns
        -------
        numpy.ndarray[int]
            The rank assigned to each sample.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state = check_random_state(random_state)
        peaks = np.linspace(0, self.n_classes, num=n_ranks + 1).round().astype(int)[:-1]
        class_weights_by_rank = {
            rank: get_skellam_class_weights(mu, self.n_classes, peak)
            for rank, peak in enumerate(peaks)
        }
        return self.partition(n_ranks, class_weights_by_rank, random_state, sampling)


def get_skellam_class_weights(
    mu: float | str,
    n_classes: int,
    peak: int = 0,
    rescale_to_sum_one: bool = True,
) -> npt.NDArray[np.float32]:
    """
    Generate Skellam-distributed class weights with a variable spread and peak.

    The spread is adjusted via the μ parameter (the larger μ, the larger the spread). Edge cases: For μ=0, the peak
    class has weight 1, while all other classes have weight 0. For μ=inf, the generated dataset is balanced, i.e., all
    classes have equal weights.

    Parameters
    ----------
    mu : float | str
        The μ = μ₁ = μ₂ parameter of the Skellam distribution. Must be ≥0 or 'inf'.
    n_classes : int
        The number of classes.
    peak : int
        The position (class index) of the distribution's peak.
    rescale_to_sum_one : bool
        Whether to rescale the weights, so they sum up to 1.

    Returns
    -------
    numpy.ndarray[float]
        The class weights.
    """
    # Edge cases
    if mu == 0:
        return np.eye(n_classes)[peak]
    elif mu == "inf" or np.isinf(float(mu)):
        return np.repeat(1 / n_classes, n_classes)

    # Standard case
    class_weight = scipy.stats.skellam.pmf(
        np.arange(n_classes), mu1=mu, mu2=mu, loc=n_classes // 2
    )
    shift = peak - n_classes // 2
    class_weight = np.roll(class_weight, shift)
    if rescale_to_sum_one:
        class_weight /= class_weight.sum()
    return class_weight


class SyntheticDataset:
    """
    Synthetic classification dataset.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    n_samples : int
        The number of samples.
    x : numpy.ndarray
        The input samples.
    y : numpy.ndarray[float]
        The corresponding targets.

    Methods
    -------
    get_class_frequencies()
        Get class frequency (either as absolute counts or as relative fraction).
    generate()
        Generate a synthetic classification dataset using ``sklearn.datasets.make_classification``.
    generate_with_skellam_imbalance()
        Generate a synthetic classification dataset using ``sklearn.datasets.make_classification``. The class weights
        are automatically determined via a Skellam distribution.
    train_test_split()
        Split this dataset into a train and test set using ``sklearn.model_selection.train_test_split``.
    get_local_subset()
        Partition the dataset among ``n_ranks`` ranks and select the samples assigned to this rank.
    allgather_class_frequencies()
        Compute class frequencies as array and all-gather them via the given communicator.
    plot_local_class_distributions()
        Plot the given local and global class distribution as ridge plot.
    plot_skellam_distributions()
        Plot class frequencies for different Skellam distributions as line plot.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_samples: int | None = None,
        n_classes: int | None = None,
    ) -> None:
        """
        Generate a synthetic classification dataset using ``sklearn.datasets.make_classification``.

        Parameters
        ----------
        x : numpy.ndarray
            The input samples x.
        y : numpy.ndarray
            The corresponding targets y.
        n_samples : int, optional
            The number of samples in the dataset.
        n_classes : int, optional
            The number of (unique) classes in the dataset.
        """
        self.x = x
        self.y = y
        self.n_samples = len(self.y) if n_samples is None else n_samples
        self.n_classes = (
            len(np.unique(self.y, axis=0)) if n_classes is None else n_classes
        )

    def __eq__(self, other: SyntheticDataset) -> bool:
        """
        Compare this dataset to another dataset and return whether they are equal. Two datasets are considered
        equal if their n_samples, n_classes, and (element-wise) y are equal and their (element-wise) features x are
        allmost equal (rtol=1e-05 and atol=1e-08).

        Parameters
        ----------
        other : SyntheticDataset

        Returns
        -------
        bool
            Whether this and the other dataset are the same.
        """
        if self.n_samples != other.n_samples:
            log.info(f"SyntheticDatasets not equal: {self.n_samples=} != {other.n_samples=}")
            return False

        if self.n_classes != other.n_classes:
            log.info(f"SyntheticDatasets not equal: {self.n_classes=} != {other.n_classes=}")
            return False

        if not (self.y == other.y).all():
            log.info(f"SyntheticDatasets not equal: self.y != other.y")
            return False

        if not np.allclose(self.x, other.x, equal_nan=True):
            log.info(f"SyntheticDatasets not equal: self.x !≈ other.x")
            return False
        return True

    def __str__(self) -> str:
        """
        Convert the ``SyntheticDataset`` to a summary containing its size, number of classes, and class frequency.

        Returns
        -------
        str
            Summary of the dataset.
        """
        histogram = ", ".join(
            [
                f"{target}: {frequency:4.2f}"
                for target, frequency in self.get_class_frequency(True).items()
            ]
        )
        return f"SyntheticDataset({self.n_samples} samples, classes: {histogram})"

    def get_class_frequency(self, relative: bool = False) -> dict[int, float]:
        """
        Get class frequency (either as absolute counts or as relative fraction).

        Parameters
        ----------
        relative : bool
            Whether to return the absolute or relative class frequency.

        Returns
        -------
        dict[int, float]
            A dict mapping each class to its frequency in y.
        """
        classes, frequencies = np.unique(self.y, return_counts=True)
        if relative:
            frequencies = frequencies / frequencies.sum()
        return dict(zip(classes, frequencies))

    @classmethod
    def generate(
        cls: Type["SyntheticDataset"],
        n_samples: int,
        n_features: int,
        n_classes: int,
        class_weights: npt.NDArray[np.float32] | None = None,
        random_state: int | np.random.RandomState | None = None,
        make_classification_kwargs: dict[str, Any] | None = None,
    ) -> "SyntheticDataset":
        """
        Generate a synthetic classification dataset using ``sklearn.datasets.make_classification``.

        Parameters
        ----------
        n_samples : int
            The number of samples in the dataset.
        n_features : int
            The number of features in the dataset.
        n_classes : int
            The number classes in the dataset.
        class_weights : numpy.ndarray, optional
            The weight for each class, default: balanced dataset, i.e., all classes have equal weight.
        random_state : int | np.random.RandomState, optional
            The random state used for the generation and distributed assignment.
        make_classification_kwargs : dict[str, Any], optional
            Additional keyword arguments to ``sklearn.datasets.make_classification``.

        Returns
        -------
        SyntheticDataset
            The generated dataset as new instance of ``SyntheticDataset``.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state = check_random_state(random_state)
        log.debug(f"Classification kwargs: {make_classification_kwargs}")
        log.debug(
            f"The random state is before generating the dataset is:\n{random_state.get_state(legacy=True)}"
        )
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            weights=class_weights,
            random_state=random_state,
            **make_classification_kwargs,
        )
        log.debug(f"First sample is:\n{x[0]}\nLast sample is:\n{x[-1]}\n")
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        return SyntheticDataset(x, y, n_samples, n_classes)

    @classmethod
    def generate_with_skellam_class_imbalance(
        cls: Type["SyntheticDataset"],
        n_samples: int,
        n_features: int,
        n_classes: int,
        mu: float | str,
        peak: int = 0,
        rescale_to_sum_one: bool = True,
        random_state: int | np.random.RandomState | None = None,
        make_classification_kwargs: dict[str, Any] | None = None,
    ) -> "SyntheticDataset":
        """
        Generate a synthetic classification dataset using ``sklearn.datasets.make_classification``.

        The class weights are automatically determined via a Skellam distribution.

        Parameters
        ----------
        n_samples : int
            The number of samples in the dataset.
        n_features : int
            The number of features in the dataset.
        n_classes : int
            The number classes in the dataset.
        mu : float | str
            The μ = μ₁ = μ₂ parameter of the Skellam distribution. Must be ≥0 or 'inf'.
        peak : int
            The position (class index) of the distribution's peak.
        rescale_to_sum_one : bool
            Whether to rescale the weights, so they sum up to 1.
        random_state : int | np.random.RandomState, optional
            The random state used for the generation and distributed assignment.
        make_classification_kwargs : dict[str, Any], optional
            Additional keyword arguments to ``sklearn.datasets.make_classification``.

        Returns
        -------
        SyntheticDataset
            The generated dataset as new instance of ``SyntheticDataset``.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state = check_random_state(random_state)
        class_weights = get_skellam_class_weights(
            mu, n_classes, peak, rescale_to_sum_one
        )
        return cls.generate(
            n_samples,
            n_features,
            n_classes,
            class_weights=class_weights,
            random_state=random_state,
            make_classification_kwargs=make_classification_kwargs,
        )

    def train_test_split(
        self,
        test_size: float,
        stratify: bool = True,
        random_state: int | np.random.RandomState | None = None,
        **kwargs: Any,
    ) -> tuple["SyntheticDataset", "SyntheticDataset"]:
        """
        Split this dataset into a train and test set using ``sklearn.model_selection.train_test_split``.

        The respective train and test sets are returned as new instances of ``SyntheticDataset``.

        Parameters
        ----------
        test_size : float
            Relative size of the test set.
        stratify : bool
            Whether to stratify the split using the class labels.
        random_state : int | np.random.RandomState, optional
            Random state for ``sklearn.model_selection.train_test_split``.
        **kwargs : Any
            Additional keyword arguments to ``sklearn.model_selection.train_test_split``.

        Returns
        -------
        SyntheticDataset
            The train set.
        SyntheticDataset
            The test set.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state = check_random_state(random_state)
        log.debug(f"Stratify is {stratify}.")
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            test_size=test_size,
            stratify=self.y if stratify else None,
            random_state=random_state,
            **kwargs,
        )
        return SyntheticDataset(
            x_train, y_train, None, self.n_classes
        ), SyntheticDataset(x_test, y_test, None, self.n_classes)

    def get_local_subset(
        self,
        rank: int,
        n_ranks: int,
        random_state: int | np.random.RandomState,
        balanced: bool,
        mu: float | str | None = None,
        sampling: bool = False,
    ) -> "SyntheticDataset":
        """
        Partition the dataset among ``n_ranks`` ranks and select the samples assigned to this rank.

        Parameters
        ----------
        rank : int
            The index of this rank.
        n_ranks : int
            The total number of ranks.
        random_state : int | np.random.RandomState
            The random state.
        balanced : bool
            Whether to use a balanced partition, if False, ``mu`` must be specified.
        mu : float | str, optional
            The μ = μ₁ = μ₂ parameter of the Skellam distribution when using an imbalanced class distribution.
        sampling : bool
            Whether to partition the dataset using deterministic element counts and shuffling (number of elements per
            rank and class independent of random seed, only which samples are assigned to which rank may change) or
            random sampling (number of elements per rank and class is expected to be the corresponding weight but may
            change slightly with each random seed).

        Returns
        -------
        SyntheticDataset
            A ``SyntheticDataset`` containing only the samples assigned to the specified rank.
        """
        # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
        random_state = check_random_state(random_state)
        partition = DatasetPartition(self.y)
        if balanced:
            assigned_ranks = partition.balanced_partition(
                n_ranks, random_state, sampling
            )
        else:
            assert mu is not None
            assigned_ranks = partition.shifted_skellam_imbalanced_partition(
                n_ranks, mu, random_state, sampling
            )
        assigned_indices = partition.assigned_indices_by_rank(assigned_ranks)[rank]
        return SyntheticDataset(
            self.x[assigned_indices], self.y[assigned_indices], None, self.n_classes
        )

    def allgather_class_frequencies(
        self, comm: MPI.Comm, relative: bool = False
    ) -> np.ndarray:
        """
        Compute class frequencies as array and all-gather them via the given communicator.

        Parameters
        ----------
        comm : MPI.Comm
            The communicator to all-gather the results in.
        relative : bool
            Whether the absolute or relative class frequencies shall be all-gathered.

        Returns
        -------
        np.ndarray
            The computed class frequencies.
        """
        local_class_frequencies = np.array(
            [
                self.get_class_frequency(relative).get(class_id, 0)
                for class_id in range(self.n_classes)
            ]
        )
        return np.array(comm.allgather(local_class_frequencies))

    @staticmethod
    def plot_local_class_distributions(
        local_class_frequencies: np.ndarray, **kwargs: Any
    ) -> tuple:
        """
        Plot the given local and global class distribution as ridge plot.

        Parameters
        ----------
        local_class_frequencies : numpy.ndarray
            The local class frequencies (ranks on axis 0, classes on axis 1).
        **kwargs : Any
            Optional additional parameters to ``utils.plot_local_class_distributions_as_ridgeplot``.

        Returns
        -------
        plt.Figure
            The figure.
        Collection[plt.Axes]
            The axis.
        """
        return plot_local_class_distributions_as_ridgeplot(
            local_class_frequencies, **kwargs
        )

    @staticmethod
    def plot_skellam_distributions(
        n_classes: int,
        mus: list[float | str],
        peaks: list[int] | None = None,
    ) -> tuple:
        """
        Plot class frequencies for different Skellam distributions as line plot.

        One line is plotted for each combination of mu and peak values.

        Parameters
        ----------
        n_classes : int
            The number of classes in the generated distributions.
        mus : list[float | str]
            A list of mu values to plot the skellam distributions for.
        peaks : list[int], optional
            An list of peaks values to plot the skellam distributions for. By default, all distributions use the center
            class as peak.

        Returns
        -------
        plt.Figure
            The figure.
        Collection[plt.Axes]
            The axis.
        """
        peaks = peaks or [n_classes // 2]

        def label(mu: float | str, peak: int) -> str:
            """Get a label for the given mu and peak."""
            if len(mus) <= 1:
                return f"peak={peak}"
            elif len(peaks) <= 1:
                return f"μ={mu:g}"
            return f"μ={mu:g}, peak={peak}"

        class_frequencies_by_mu = {
            label(mu, peak): get_skellam_class_weights(mu, n_classes, peak)
            for mu, peak in itertools.product(mus, peaks)
        }
        return plot_class_distributions(class_frequencies_by_mu, title="")


def generate_scaling_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_ranks: int,
    random_state: int | np.random.RandomState,
    test_size: float,
    make_classification_kwargs: dict[str, Any] | None = None,
    sampling: bool = False,
    stratified_train_test: bool = False,
    rank: int | None = None,
) -> tuple[SyntheticDataset, dict[int, SyntheticDataset] | SyntheticDataset, SyntheticDataset]:
    """
    Generate a dataset to be scaled with the number of nodes. Generates a single global dataset and splits it into
    n_ranks slices. Returns the global train and test set and either the local train set of the given rank, or a dict of
    all local train sets (mapping rank -> local train set) when rank is None.

    Parameters
    ----------
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number classes in the dataset.
    n_ranks : int
        The total/maximum number of ranks to generate the dataset for.
    random_state : int | np.random.RandomState
        The random state, used for dataset generation, partition, and distribution.
    test_size : float
        Relative size of the test set.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    sampling : bool
        Whether to partition the dataset using deterministic element counts and shuffling or random sampling.
        See ``SyntheticDataset.get_local_subset`` for more details.
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels.
    rank : int | None
        When given, returns the local train set for this rank. Otherwise (i.e. for None), a dict of all local train sets
        is returned.

    Returns
    -------
    SyntheticDataset
        The global dataset.
    dict[int, SyntheticDataset] | SyntheticDataset
        Either the local train subset for this rank (if given) or a dict of all local train sets by rank.
    SyntheticDataset
        The global = local test set.

    """
    random_state = check_random_state(random_state)

    log.debug(f"The random state is:\n{random_state.get_state(legacy=True)}")
    log.debug(f"Classification kwargs: {make_classification_kwargs}")

    # Step 1: generate balanced global dataset for up to n_ranks nodes
    global_dataset = SyntheticDataset.generate(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
        make_classification_kwargs=make_classification_kwargs,
    )

    # Step 2: split into global train and test set
    # TODO: in this case, the size of the test set scales with n_ranks -> do we want this? if not, what else?
    log.debug(f"Generate global train-test split: {test_size=}, {stratified_train_test=}.")
    global_train_set, global_test_set = global_dataset.train_test_split(
        test_size=test_size,
        stratify=stratified_train_test,
        random_state=random_state,
    )
    log.debug(f"Shape of global train set {global_train_set.x.shape}, test set {global_test_set.x.shape}")

    # Step 3: partition the global train set into n_ranks local train sets (balanced partition)
    partition = DatasetPartition(global_train_set.y)
    assigned_ranks = partition.balanced_partition(n_ranks, random_state, sampling)
    assigned_indices = partition.assigned_indices_by_rank(assigned_ranks)
    training_slices = {
        rank: SyntheticDataset(global_train_set.x[indices], global_train_set.y[indices], None, n_classes)
        for rank, indices in assigned_indices.items()
    }

    log.debug(f"Shape of local train set 0 {training_slices[0].x.shape}")

    if rank is not None:
        log.debug(f"Returning local train set for rank {rank}")
        return global_train_set, training_slices[rank], global_test_set
    else:
        log.debug(f"Returning dict of all local train sets")
        return global_train_set, training_slices, global_test_set


def write_scaling_dataset_to_hdf5(
    global_train_set: SyntheticDataset,
    local_train_sets: dict[int, SyntheticDataset],
    global_test_set: SyntheticDataset,
    additional_global_attrs: dict[str, Any],
    file_path: os.PathLike,
    override: bool = False,
) -> None:
    """
    Write a scaling dataset as HDF5 file to the given path. The HDF5 file has the following structure:
    Root Attributes:
    - '/n_classes': the number of classes in the dataset
    - '/n_ranks': the number of ranks into which the dataset has been partitioned
    - '/n_samples_global_train': the number of samples in the global training set
    - **additional_global_attrs: additional_global_attrs are written directly to the root attributes
      can for example be used to store the config used to generate this dataset

    Groups:
    - '/test_set': for the global test_set
    - '/local_train_sets/rank_{rank}': the local train set for each rank

    Group Attributes:
    - 'n_samples': samples in this subset
    - 'label': label of the subset (= group name)
    - 'rank': assigned rank as int (only for local train sets)


    Parameters
    ----------
    global_train_set : SyntheticDataset
        The global train set (used only to extract root attributes)
    local_train_sets : dict[int, SyntheticDataset]
        pass
    global_test_set : SyntheticDataset
        pass
    additional_global_attrs : dict[str, Any]
        pass
    file_path : os.PathLike
        The file path of the HDF5 file to write.
    override : bool
        If true, will override any existing files at file_path. If false, will raise a FileExistsError
        should file_path already exist.
    """
    file_path = pathlib.Path(file_path)
    if not override and file_path.exists():
        raise FileExistsError(f"File {file_path} exists and override is set to {file_path}.")
    file = h5py.File(file_path, "w")

    file.attrs["n_classes"] = global_train_set.n_classes
    file.attrs["n_ranks"] = len(local_train_sets)
    file.attrs["n_samples_global_train"] = global_train_set.n_samples
    for key, value in additional_global_attrs.items():
        file.attrs[key] = value

    def write_subset_to_group(group_name, dataset, **attrs):
        file[f'{group_name}/x'] = dataset.x
        file[f'{group_name}/y'] = dataset.y
        file[group_name].attrs["n_samples"] = dataset.n_samples
        for key, value in attrs.items():
            file[group_name].attrs[key] = value

    # write global test set to HDF5
    write_subset_to_group("test_set", global_test_set, label="global_test_set")

    # write local train sets to HDF5
    for rank, local_train_set in local_train_sets.items():
        group_name = f"local_train_sets/rank_{rank}"
        write_subset_to_group(group_name, local_train_set, label=group_name, rank=rank)


def read_scaling_dataset_from_hdf5(
    file_path: os.PathLike,
    rank: int | None = None,
) -> tuple[dict[int, SyntheticDataset] | SyntheticDataset, SyntheticDataset, dict[str, Any]]:
    """
    Read a scaling dataset (local train sets and global test set) from the given HDF5 file.

    Parameters
    ----------
    file_path : os.PathLike
        Path to the HDF5 file to read.
    rank : int | None
        Optional rank. When given, only the local train set for the specified rank is retrieved. Otherwise, all local
        train sets are retrieved.

    Returns
    -------
    dict[int, SyntheticDataset] | SyntheticDataset
        A dict of all local train sets by rank or just the local train set for the given rank.
    SyntheticDataset
        The global = local test set.
    dict[str, Any]
        Any additional information on the dataset stored as root attribute in the HDF5 file.
    """
    file = h5py.File(file_path, "r")
    n_classes = file.attrs["n_classes"]
    root_attrs = dict(file.attrs)

    def dataset_from_group(group):
        return SyntheticDataset(
            group["x"], group["y"], n_samples=int(group.attrs["n_samples"]), n_classes=int(n_classes)
        )

    global_test_set = dataset_from_group(file['test_set'])

    if rank is None:  # dict of all local train sets
        local_train_set = {group.attrs['rank']: dataset_from_group(group)
                           for name, group in file['local_train_sets'].items()}
    else:  # just the local train set for the given rank
        local_train_set = dataset_from_group(file[f'local_train_sets/rank_{rank}'])

    return local_train_set, global_test_set, root_attrs


def generate_and_distribute_synthetic_dataset(
    globally_balanced: bool,
    locally_balanced: bool,
    n_samples: int,
    n_features: int,
    n_classes: int,
    rank: int,
    n_ranks: int,
    random_state: int | np.random.RandomState,
    test_size: float,
    mu_partition: float | str | None = None,
    mu_data: float | str | None = None,
    peak: int | None = None,
    rescale_to_sum_one: bool = True,
    make_classification_kwargs: dict[str, Any] | None = None,
    sampling: bool = False,
    shared_test_set: bool = True,
    stratified_train_test: bool = False,
) -> tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
    """
    Generate a synthetic dataset, partition it among all ranks, and determine the subset assigned to this rank.

    Parameters
    ----------
    globally_balanced : bool
        Whether the class distribution of the entire dataset is balanced. If False, ``mu_data`` must be specified.
    locally_balanced : bool
        Whether to use a balanced partition when assigning the dataset to ranks. If False, ``mu_partition`` must be
        specified.
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number classes in the dataset.
    rank : int
        The index of this rank.
    n_ranks : int
        The total number of ranks.
    random_state : int | np.random.RandomState
        The random state, used for dataset generation, partition, and distribution.
    test_size : float
        Relative size of the test set.
    mu_partition : float | str, optional
        The μ parameter of the Skellam distribution for imbalanced class distribution. Has no effect if
        ``locally_balanced`` is True.
    mu_data : float | str, optional
        The μ parameter of the Skellam distribution for imbalanced class distribution in the dataset. Has no effect if
        ``globally_balanced`` is True.
    peak : int, optional
        The position (class index) of the class distribution peak in the dataset. Has no effect if ``globally_balanced``
        is True.
    rescale_to_sum_one : bool
        Whether to rescale the class weights, so they sum up to 1. Default is True.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    sampling : bool
        Whether to partition the dataset using deterministic element counts and shuffling or random sampling.
        See ``SyntheticDataset.get_local_subset`` for more details.
    shared_test_set : bool
        Whether the test set is shared among all ranks or each rank has its own test set that is not shared with the
        other ranks. In practice, this decides whether we first split the data into train-test and then distribute
        the train set (shared test set) or first distribute the dataset and then split the local subsets into train
        and test (private test sets).
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels.

    Returns
    -------
    SyntheticDataset
        The global dataset.
    SyntheticDataset
        The local train subset containing only the samples assigned to this rank.
    SyntheticDataset
        The local test subset containing only the samples assigned to this rank.
    """
    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    random_state = check_random_state(random_state)
    log.debug(f"The random state is:\n{random_state.get_state(legacy=True)}")
    # Generate dataset.
    log.debug(f"Classification kwargs: {make_classification_kwargs}")
    if globally_balanced:
        global_dataset = SyntheticDataset.generate(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            random_state=random_state,
            make_classification_kwargs=make_classification_kwargs,
        )

        log.debug(
            f"First sample is:\n{global_dataset.x[0]}\nLast sample is:\n{global_dataset.x[-1]}"
        )
    else:
        assert mu_data is not None and peak is not None
        global_dataset = SyntheticDataset.generate_with_skellam_class_imbalance(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            mu=mu_data,
            peak=peak,
            rescale_to_sum_one=rescale_to_sum_one,
            random_state=random_state,
            make_classification_kwargs=make_classification_kwargs,
        )

    if shared_test_set:  # Case 1: Shared test set, all ranks use the same test set.
        log.debug("Generate global train-test split.")
        log.debug(f"Stratify train-test split? {stratified_train_test}")
        # First: Train-test split
        global_train_set, global_test_set = global_dataset.train_test_split(
            test_size=test_size,
            stratify=stratified_train_test,
            random_state=random_state,
        )
        log.debug(
            f"First global train sample is:\n{global_train_set.x[0]}\nLast train sample is:\n{global_train_set.x[-1]}\n"
            f""
            f"First global test sample is:\n{global_test_set.x[0]}\nLast test sample is:\n{global_test_set.x[-1]}"
        )

        log.debug(
            f"Global train set shape: {global_train_set.x.shape}: Global shared test set shape: {global_test_set.x.shape}"
        )
        local_test = global_test_set  # in this case, the local test set is the same as the global test set
        # Then: Partition only the training data and distribute them across ranks.
        local_train = global_train_set.get_local_subset(
            rank,
            n_ranks,
            random_state,
            balanced=locally_balanced,
            mu=mu_partition,
            sampling=sampling,
        )
        log.debug(f"Local train set shape: {local_train.x.shape}")
    else:  # Case 2: Private test set, each rank has its own test set that is not shared with the other ranks.
        # First: Partition the entire dataset.
        log.debug("Generate private train-test split.")
        local_subset = global_dataset.get_local_subset(
            rank,
            n_ranks,
            random_state,
            balanced=locally_balanced,
            mu=mu_partition,
            sampling=sampling,
        )
        # Then: Perform train-test splits locally on each rank.
        local_train, local_test = local_subset.train_test_split(
            test_size=test_size,
            stratify=stratified_train_test,
            random_state=random_state,
        )

    return global_dataset, local_train, local_test


def data_generation_demo(
    globally_balanced: bool,
    locally_balanced: bool,
    shared_test_set: bool,
    n_samples: int = 100,
    n_features: int = 10,
    n_classes: int = 5,
    n_ranks: int = 4,
    random_state: int | np.random.RandomState = 0,
    mu_partition: float | str = 2,
    mu_data: float | str = 2,
    peak: int = 0,
    test_size: float = 0.2,
    path: pathlib.Path | str | None = None,
) -> None:
    """
    Demonstrate how to generate a synthetic dataset and partition it among all ranks.

    Parameters
    ----------
    globally_balanced : bool
        Whether the generated dataset is globally balanced (True) or imbalanced (False).
    locally_balanced : bool
        Whether the generated dataset is locally balanced (True) or imbalanced (False).
    shared_test_set : bool
        Whether all ranks share the same test set (True) or each of them uses its own private test set (False).
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number classes in the dataset.
    n_ranks : int
        The total number of ranks, i.e., processes, in the communicator.
    random_state : int | np.random.RandomState
        The random state, used for both the dataset generation and the partition and distribution.
    mu_partition : float | str, optional
        The μ parameter of the Skellam distribution for imbalanced class distribution. Has no effect if
        ``locally_balanced`` is True.
    mu_data : float | str, optional
        The μ parameter of the Skellam distribution for imbalanced class distribution in the dataset. Has no effect if
        ``globally_balanced`` is True.
    peak : int, optional
        The position (class index) of the class distribution peak in the dataset. Has no effect if ``globally_balanced``
        is True.
    test_size : float
        Relative size of the test set.
    path : pathlib.Path | str, optional
        An optional output path to store result figures to.
    """
    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    random_state = check_random_state(random_state)
    path = pathlib.Path(path) if path is not None else None
    log.info(f"\n{globally_balanced=}, {locally_balanced=}, {shared_test_set=}")
    label = f"{shared_test_set=}__{globally_balanced=}__{locally_balanced=}"
    local_datasets: dict[str, list] = {"train": [], "test": []}

    for rank in range(n_ranks):
        log.info(f"---- Rank {rank} ------------------------")
        (
            global_dataset,
            local_train,
            local_test,
        ) = generate_and_distribute_synthetic_dataset(
            globally_balanced=globally_balanced,
            locally_balanced=locally_balanced,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            rank=rank,
            n_ranks=n_ranks,
            random_state=random_state,
            test_size=test_size,
            mu_partition=mu_partition,
            mu_data=mu_data,
            peak=peak,
            shared_test_set=shared_test_set,
        )

        log.info(
            f"Global: {global_dataset}\n Train: {local_train}\n  Test: {local_test}\n"
        )

        local_datasets["train"].append(local_train)
        local_datasets["test"].append(local_test)

    if path:
        local_class_frequencies = {
            key: np.array(
                [
                    [
                        dataset.get_class_frequency(True).get(class_id, 0)
                        for class_id in range(n_classes)
                    ]
                    for dataset in datasets
                ]
            )
            for key, datasets in local_datasets.items()
        }

        for key, class_frequency in local_class_frequencies.items():
            class_distribution_fig, _ = SyntheticDataset.plot_local_class_distributions(
                class_frequency, figsize=[3, 3]
            )
            class_distribution_fig.savefig(
                path / f"class_distribution__{label}__{key}.pdf"
            )
            plt.close(class_distribution_fig)


def make_classification_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int = 10,
    make_classification_kwargs: dict[str, Any] | None = None,
    random_state: int | np.random.RandomState | None = 0,
    train_split: float = 0.75,
    stratified_train_test: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate globally balanced synthetic classification dataset for non-distributed case.

    Parameters
    ----------
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_classes : int
        The number of classes. Default is 10.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    random_state : int | np.random.RandomState, optional
        The random state for dataset generation and splitting. Default is 0.
    train_split : float
        The train-test split fraction. Default is 0.75.
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels. Default is False.

    Returns
    -------
    numpy.ndarray
        The train samples.
    numpy.ndarray
        The train targets.
    numpy.ndarray
        The test samples.
    numpy.ndarray
        The test targets.
    """
    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    random_state = check_random_state(random_state)
    log.debug(
        f"Random state before generating the dataset is:\n{random_state.get_state(legacy=True)}\n"
        f"`make_classification_kwargs`:\n{make_classification_kwargs}"
    )

    # Generate data as numpy arrays.
    samples, targets = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
        **make_classification_kwargs,
    )
    log.debug(f"First sample is:\n{samples[0]}\nLast sample is:\n{samples[-1]}")

    targets = targets.astype(np.float32)
    samples = samples.astype(np.float32)
    # Split into train and test set.
    return train_test_split(
        samples,
        targets,
        test_size=1 - train_split,
        stratify=targets if stratified_train_test else None,
        random_state=random_state,
    )


def make_classification_dataset_no_split(
    n_samples: int,
    n_features: int,
    n_classes: int = 10,
    make_classification_kwargs: dict[str, Any] | None = None,
    random_state: int | np.random.RandomState | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate globally balanced synthetic classification dataset for non-distributed case.

    Parameters
    ----------
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_classes : int
        The number of classes. Default is 10.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    random_state : int | np.random.RandomState, optional
        The random state for dataset generation and splitting. Default is 0.

    Returns
    -------
    numpy.ndarray
        The samples.
    numpy.ndarray
        The targets.
    """
    # Check passed random state and convert if necessary, i.e., turn into a ``np.random.RandomState`` instance.
    random_state = check_random_state(random_state)
    log.debug(
        f"Random state before generating the dataset is:\n{random_state.get_state(legacy=True)}\n"
        f"`make_classification_kwargs`:\n{make_classification_kwargs}"
    )

    # Generate data as numpy arrays.
    samples, targets = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
        **make_classification_kwargs,
    )
    log.debug(f"First sample is:\n{samples[0]}\nLast sample is:\n{samples[-1]}")
    return samples.astype(np.float32), targets.astype(np.float32)


if __name__ == "__main__":
    output_path = pathlib.Path(__file__).parent.parent / "results" / "breaking_iid_demo"
    output_path.mkdir(parents=True, exist_ok=True)

    for globally_balanced, locally_balanced, shared_test_set in itertools.product(
        *[[True, False]] * 3
    ):
        data_generation_demo(
            globally_balanced=globally_balanced,
            locally_balanced=locally_balanced,
            shared_test_set=shared_test_set,
            n_samples=10000,
            n_classes=11,
            mu_partition=5,
            mu_data=5,
            peak=5,
            path=output_path,
        )

    # impact of mu
    fig, _ = SyntheticDataset.plot_skellam_distributions(
        11, mus=[0, 0.1, 0.5, 1, 2, 5, float("inf")]
    )
    fig.savefig(output_path / "skellam_distributions.svg")
