import collections
import itertools
import pathlib
from typing import Dict, Callable, Optional, Union, Type, Tuple, List

import numpy as np
import scipy
from mpi4py import MPI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import utils


class DatasetPartition:
    def __init__(self, targets: np.ndarray) -> None:
        """
        Initialize the partition with the targets for each sample, automatically determines a list of sample indices for
        each class.

        Parameters
        ----------
        targets : numpy.ndarray
            A 1D numpy array containing the target for each sample.
        """
        self.targets = targets
        self.num_samples = len(self.targets)
        self.indices_by_class = {
            class_index: np.nonzero(self.targets == class_index)[0]
            for class_index in np.unique(self.targets)
        }
        self.num_classes = len(self.indices_by_class.keys())

    def get_class_imbalance(self) -> Dict[int, float]:
        """
        Get the percentage share for each target value from an array of targets.

        Returns
        -------
        Dict
            A dict mapping each target value to its percentage.
        """
        unique, counts = np.unique(self.targets, return_counts=True)
        counts = counts / counts.sum()
        return dict(zip(unique, counts))

    @staticmethod
    def assigned_indices_by_rank(
        assigned_ranks: np.ndarray[int],
    ) -> Dict[int, np.ndarray[int]]:
        """
        Turns an array assigning each sample index to a rank into a dict mapping each rank to its assigned sample
        indices (as array).

        Parameters
        ----------
        assigned_ranks : numpy.ndarray[int]
            Numpy array of length num_samples, assigning each sample to a rank.

        Returns
        -------
        Dict[int, numpy.ndarray[int]]
            A dict mapping each rank to the assigned sample indices.
        """
        return {
            rank: np.nonzero(assigned_ranks == rank)[0]
            for rank in np.unique(assigned_ranks)
        }

    @staticmethod
    def weighted_partition(
        total_elements: int, weights: np.ndarray[float]
    ) -> np.ndarray[int]:
        """
        Partition a number of elements into len(weights) parts according to the weights (as closely as possible).

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

        # assign remaining elements to the parts with the largest difference between actual and current weight
        remainders = percentage * total_elements - element_count_per_part
        remaining_elements = total_elements - element_count_per_part.sum()
        if remaining_elements > 0:
            index_in_increasing_weight = np.argsort(remainders)
            element_count_per_part[
                index_in_increasing_weight[-remaining_elements:]
            ] += 1

        return element_count_per_part

    def _partition_class_wise(
        self, class_partitioner: Callable[[int, np.ndarray[int]], np.ndarray[int]]
    ) -> np.ndarray[int]:
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

        # not the most performant but it's only called once
        index_rank_pairs = [
            (sample_index, rank)
            for c in assigned_ranks_by_class.keys()
            for (sample_index, rank) in zip(
                self.indices_by_class[c], assigned_ranks_by_class[c]
            )
        ]
        assigned_ranks = np.repeat(-1, self.num_samples)
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
        class_weights_by_rank: Dict[int, np.ndarray[float]]
    ) -> Dict[int, np.ndarray[float]]:
        """
        Converts a dict containing the class balance for each rank to a dict containing the rank balance for each class.

        Parameters
        ----------
        class_weights_by_rank : Dict[int, numpy.ndarray[float]]
            The class balance for each rank as dict rank -> class weights.

        Returns
        -------
        Dict[int, numpy.ndarray[float]]
            The rank balance (in percent) for each class as dict class -> rank_weights.
        """
        num_ranks = len(class_weights_by_rank)
        num_classes = len(class_weights_by_rank[0])
        rank_weights_by_class = {
            c: np.array([class_weights_by_rank[rank][c] for rank in range(num_ranks)])
            for c in range(num_classes)
        }
        # convert to probabilities and return
        return {
            c: rank_weights / rank_weights.sum()
            for c, rank_weights in rank_weights_by_class.items()
        }

    @staticmethod
    def deterministic_class_partitioner(
        num_ranks: int,
        class_weights_by_rank: Optional[Dict[int, np.ndarray[float]]] = None,
        seed: Optional[int] = None,
    ) -> Callable[[int, np.ndarray[int]], np.ndarray[int]]:
        """
        Returns a function which partitions the samples for a class based on optional class weights by calculating the
        number of samples per rank and shuffling the assignment.

        Parameters
        ----------
        num_ranks : int
            The number of ranks to partition into.
        class_weights_by_rank : Optional[Dict[int, numpy.ndarray[float]]]
            The class balance for each rank as dict rank -> class_weights. If None, a uniform class balance is assumed.
        seed : Optional[int]
            The random seed used for the shuffling.

        Returns
        -------
        Callable[[int, numpy.ndarray[int]], numpy.ndarray[int]]
            A functon mapping the class index and sample indices to the assigned rank by sample.
        """
        rng = np.random.default_rng(seed)
        if class_weights_by_rank is None:
            rank_probabilities_by_class = collections.defaultdict(
                lambda: np.repeat(1 / num_ranks, num_ranks)
            )
        else:
            rank_probabilities_by_class = (
                DatasetPartition._class_weights_to_rank_weights(class_weights_by_rank)
            )

        def assign_to_ranks(
            class_index: int, sample_indices: np.ndarray[int]
        ) -> np.ndarray[int]:
            elements_per_rank = DatasetPartition.weighted_partition(
                len(sample_indices), rank_probabilities_by_class[class_index]
            )
            assigned_ranks = np.array(
                [
                    rank
                    for rank, num_elements in enumerate(elements_per_rank)
                    for _ in range(num_elements)
                ]
            )
            rng.shuffle(assigned_ranks)
            return assigned_ranks

        return assign_to_ranks

    @staticmethod
    def sampling_class_partitioner(
        num_ranks: int,
        class_weights_by_rank: Optional[Dict[int, np.ndarray]] = None,
        seed: Optional[int] = None,
        **choice_kwargs,
    ) -> Callable[[int, np.ndarray[int]], np.ndarray[int]]:
        """
        Returns a function which partitions the samples for a class based on optional class weights by sampling a random
        rank based on its probability.

        Parameters
        ----------
        num_ranks : int
            The number of ranks to partition into.
        class_weights_by_rank : Optional[Dict[int, numpy.ndarray[float]]]
            The class balance for each rank as dict rank -> class_weights. If None, a uniform class balance is assumed.
        seed : Optional[int]
            The random seed used for the random sampling.
        choice_kwargs :
            Optional additional kwargs to numpy's rng.choice.

        Returns
        -------
        Callable[[int, numpy.ndarray[int]], ndarray[int]]
            A functon mapping the class index and sample indices to the assigned rank by sample.
        """
        rng = np.random.default_rng(seed)

        if class_weights_by_rank is None:
            rank_probabilities_by_class = collections.defaultdict(None)
        else:
            rank_probabilities_by_class = (
                DatasetPartition._class_weights_to_rank_weights(class_weights_by_rank)
            )

        def assign_to_ranks(
            class_index: int, sample_indices: np.ndarray[int]
        ) -> np.ndarray[int]:
            return rng.choice(
                num_ranks,
                len(sample_indices),
                p=rank_probabilities_by_class[class_index],
                **choice_kwargs,
            )

        return assign_to_ranks

    def partition(
        self,
        num_ranks: int,
        class_weights_by_rank: Optional[Dict[int, np.ndarray]],
        seed: Optional[int] = None,
        sampling: bool = False,
    ) -> np.ndarray[int]:
        """
        Partition this dataset among num_rank ranks according to the given class weights.

        Parameters
        ----------
        num_ranks : int
            The number of ranks to partition into.
        class_weights_by_rank : Optional[Dict[int, numpy.ndarray[float]]]
            The class balance for each rank as dict rank -> class_weights. If None, a uniform class balance is assumed.
        seed : Optional[int]
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
        if sampling:
            class_partitioner = self.sampling_class_partitioner(
                num_ranks, class_weights_by_rank, seed
            )
        else:
            class_partitioner = self.deterministic_class_partitioner(
                num_ranks, class_weights_by_rank, seed
            )
        return self._partition_class_wise(class_partitioner)

    def balanced_partition(
        self, num_ranks: int, seed: Optional[int] = None, sampling: bool = False
    ) -> np.ndarray[int]:
        """
        Balanced partition of this dataset among num_rank ranks.

        Parameters
        ----------
        num_ranks : int
            The number of ranks to partition into.
        seed : Optional[int]
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
        return self.partition(num_ranks, None, seed, sampling)

    def shifted_skellam_imbalanced_partition(
        self,
        num_ranks: int,
        mu: Union[float, str],
        seed: Optional[int] = None,
        sampling: bool = False,
    ) -> np.ndarray[int]:
        """
        Partition this dataset among num_rank ranks using an imbalanced class distribution via a shifted skellam
        distribution.

        Parameters
        ----------
        num_ranks : int
            The number of ranks to partition into.
        mu : Union[float, str]
            The μ = μ₁ = μ₂ parameter of the skellam distribution. Must be ≥0 or 'inf'
        seed : Optional[int]
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
        peaks = (
            np.linspace(0, self.num_classes, num=num_ranks + 1).round().astype(int)[:-1]
        )
        class_weights_by_rank = {
            rank: get_skellam_class_weights(mu, self.num_classes, peak)
            for rank, peak in enumerate(peaks)
        }
        return self.partition(num_ranks, class_weights_by_rank, seed, sampling)


def get_skellam_class_weights(
    mu: Union[float, str],
    num_classes: int,
    peak: int = 0,
    rescale_to_sum_one: bool = True,
) -> np.ndarray[float]:
    """
    Generate skellam distributed class weights with a variable spread adjusted via the μ parameter (the larger μ,
    the larger the spread) and peak. Edge cases: For μ=0, the peak class has weight 1, while all other classes have
    weight 0. For μ=inf, the generated dataset is balanced, i.e., all classes have equal weights.

    Parameters
    ----------
    mu : Union[float, str]
        The μ = μ₁ = μ₂ parameter of the skellam distribution. Must be ≥0 or 'inf'
    num_classes : int
        The number of classes.
    peak : int
        The position (class index) of the distribution's peak.
    rescale_to_sum_one : bool
        Whether to rescale the weights, so they sum to 1.

    Returns
    -------
    numpy.ndarray[float]
        The class weights as numpy array.
    """
    # edge cases
    if mu == 0:
        return np.eye(num_classes)[peak]
    elif mu == "inf" or np.isinf(float(mu)):
        return np.repeat(1 / num_classes, num_classes)

    # standard case
    class_weight = scipy.stats.skellam.pmf(
        np.arange(num_classes), mu1=mu, mu2=mu, loc=num_classes // 2
    )
    shift = peak - num_classes // 2
    class_weight = np.roll(class_weight, shift)
    if rescale_to_sum_one:
        class_weight /= class_weight.sum()
    return class_weight


class SyntheticDataset:
    DEFAULT_CONFIG_MAKE_CLASSIFICATION = {"flip_y": 0, "n_informative": 8}

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        num_samples: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        """
        Generates a synthetic classification dataset using sklearn.datasets.make_classification.

        Parameters
        ----------
        x : numpy.ndarray
            The input samples x.
        y : numpy.ndarray
            The corresponding targets y.
        num_samples : Optional[int]
            The number of samples in the dataset.
        num_classes : Optional[int]
            The number of (unique) classes in the dataset.
        """
        self.x = x
        self.y = y
        self.num_samples = len(self.y) if num_samples is None else num_samples
        self.num_classes = (
            len(np.unique(self.y, axis=0)) if num_classes is None else num_classes
        )

    def __str__(self) -> str:
        """
        Converts the SyntheticDataset to a summary containing the dataset size, number of classes and class frequency.

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
        return f"SyntheticDataset({self.num_samples} samples, classes: {histogram})"

    def get_class_frequency(self, relative: bool = False) -> Dict[int, float]:
        """
        Get class frequency (either as absolute counts or as relative fraction).

        Parameters
        ----------
        relative : bool
            Whether to return the absolute or relative class frequency.

        Returns
        -------
        Dict[int, float]
            A dict mapping each class to its frequency in y.
        """
        classes, frequencies = np.unique(self.y, return_counts=True)
        if relative:
            frequencies = frequencies / frequencies.sum()
        return dict(zip(classes, frequencies))

    @classmethod
    def generate(
        cls: Type["SyntheticDataset"],
        num_samples: int,
        num_classes: int,
        class_weights: Optional[np.ndarray[float]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> "SyntheticDataset":
        """
        Generate a synthetic classification dataset using sklearn.datasets.make_classification.

        Parameters
        ----------
        num_samples : int
            The number of samples in the dataset.
        num_classes : int
            The number classes in the dataset.
        class_weights : Optional[numpy.ndarray]
            The weight for each class, default: balanced dataset, i.e., all classes have equal weight.
        random_state : Optional[int]
            The random state used for the generation and distributed assignment.
        kwargs
            Additional kwargs to sklearn.datasets.make_classification.

        Returns
        -------
        SyntheticDataset
            The generated dataset as new instance of SyntheticDataset.
        """
        make_classification_kwargs = {
            **cls.DEFAULT_CONFIG_MAKE_CLASSIFICATION,
            **kwargs,
        }
        x, y = make_classification(
            n_samples=num_samples,
            n_classes=num_classes,
            weights=class_weights,
            random_state=random_state,
            **make_classification_kwargs,
        )
        return SyntheticDataset(x, y, num_samples, num_classes)

    @classmethod
    def generate_with_skellam_class_imbalance(
        cls: Type["SyntheticDataset"],
        num_samples: int,
        num_classes: int,
        mu: Union[float, str],
        peak: int = 0,
        rescale_to_sum_one: bool = True,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> "SyntheticDataset":
        """
        Generates a synthetic classification dataset using sklearn.datasets.make_classification.
        The class weights are automatically determined via a skellam distribution.

        Parameters
        ----------
        num_samples : int
            The number of samples in the dataset.
        num_classes : int
            The number classes in the dataset.
        mu : Union[float, str]
            The μ = μ₁ = μ₂ parameter of the skellam distribution. Must be ≥0 or 'inf'
        peak : int
            The position (class index) of the distribution's peak.
        rescale_to_sum_one : bool
            Whether to rescale the weights, so they sum to 1.
        random_state : int
            The random state used for the generation and distributed assignment.
        kwargs
            Additional kwargs to sklearn.datasets.make_classification.

        Returns
        -------
        SyntheticDataset
            The generated dataset as new instance of SyntheticDataset.
        """
        class_weights = get_skellam_class_weights(
            mu, num_classes, peak, rescale_to_sum_one
        )
        return cls.generate(
            num_samples,
            num_classes,
            class_weights=class_weights,
            random_state=random_state,
            **kwargs,
        )

    def train_test_split(
        self,
        test_size: float,
        stratify: bool = True,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Tuple["SyntheticDataset", "SyntheticDataset"]:
        """
        Split this dataset into a train and test set using sklearn.model_selection.train_test_split and return them as
        new SyntheticDatasets

        Parameters
        ----------
        test_size : float
            Relative size of the test set.
        stratify : bool
            Whether to stratify the split using the class labels.
        random_state : Optional[int]
            Random seed for sklearn.model_selection.train_test_split.
        kwargs
            Additional arguments to sklearn.model_selection.train_test_split.

        Returns
        -------
        Tuple[SyntheticDataset, SyntheticDataset]
            A tuple containing the train and test set.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            test_size=test_size,
            stratify=self.y if stratify else None,
            random_state=random_state,
            **kwargs,
        )
        train = SyntheticDataset(x_train, y_train, None, self.num_classes)
        test = SyntheticDataset(x_test, y_test, None, self.num_classes)
        return train, test

    def get_local_subset(
        self,
        rank: int,
        num_ranks: int,
        seed: int,
        balanced: bool,
        mu: Union[float, str] = None,
        sampling: bool = False,
    ) -> "SyntheticDataset":
        """
        Partition the dataset among num_ranks ranks and select the samples assigned to this rank.

        Parameters
        ----------
        rank : int
            The index of this rank.
        num_ranks : int
            The total number of ranks.
        seed : int
            The random seed.
        balanced : bool
            Whether to use a balanced partition, if False, mu must be specified.
        mu : Union[float, str]
            The μ = μ₁ = μ₂ parameter of the skellam distribution when using an imbalanced class distribution.
        sampling : bool
            Whether to partition the dataset using deterministic element counts and shuffling (number of elements per
            rank and class independent of random seed, only which samples are assigned to which rank may change) or
            random sampling (number of elements per rank and class is expected to be the corresponding weight but may
            change slightly with each random seed).

        Returns
        -------
        SyntheticDataset
            A SyntheticDataset containing only the samples assigned to the specified rank.
        """
        partition = DatasetPartition(self.y)
        if balanced:
            assigned_ranks = partition.balanced_partition(num_ranks, seed, sampling)
        else:
            assert mu is not None
            assigned_ranks = partition.shifted_skellam_imbalanced_partition(
                num_ranks, mu, seed, sampling
            )
        assigned_indices = partition.assigned_indices_by_rank(assigned_ranks)[rank]
        return SyntheticDataset(
            self.x[assigned_indices], self.y[assigned_indices], None, self.num_classes
        )

    def gather_class_frequencies(
        self, comm: MPI.Comm, root: int = 0, relative: bool = False
    ) -> Union[None, np.ndarray]:
        """
        Compute class frequencies as array and gather them via the given communicator.

        Parameters
        ----------
        comm : MPI.Comm
            The communicator to gather the results in.
        root : int
            The root rank to gather the results.
        relative : bool
            Whether the absolute or relative class frequencies shall be gathered.

        Returns
        -------
        Union[None, np.ndarray]
            The computed class frequencies on the root rank, None on all other ranks.
        """
        local_class_frequencies = np.array(
            [
                self.get_class_frequency(relative).get(class_id, 0)
                for class_id in range(self.num_classes)
            ]
        )
        return np.array(comm.gather(local_class_frequencies, root=root))

    @staticmethod
    def plot_local_class_distributions(
        local_class_frequencies: np.ndarray, **kwargs
    ) -> Tuple:
        """
        Plot the given local and global class distribution as ridgeplot.

        Parameters
        ----------
        local_class_frequencies : numpy.array
            The local class frequencies (ranks on axis 0, classes on axis 1).
        kwargs : 
            Optional additional parameters to utils.plot_local_class_distributions_as_ridgeplot

        Returns
        -------
        Tuple[plt.Figure, Collection[plt.Axes]]
            The figure and axis.
        """
        return utils.plot_local_class_distributions_as_ridgeplot(
            local_class_frequencies, **kwargs
        )

    @staticmethod
    def plot_skellam_distributions(
        num_classes: int,
        mus: List[Union[float, str]],
        peaks: Optional[List[int]] = None,
    ) -> Tuple:
        """
        Plot class frequencies for different skellam distributions as line plot. One line is plotted for each
        combination of mu and peak values.

        Parameters
        ----------
        num_classes : int
            The number of classes in the generated distributions.
        mus : List[Union[float, str]]
            A list of mu values to plot the skellam distributions for.
        peaks : Optional[List[int]]
            An list of peaks values to plot the skellam distributions for. By default, all distributions use the center
            class as peak.

        Returns
        -------
        Tuple[plt.Figure, Collection[plt.Axes]]
            The figure and axis.
        """
        peaks = peaks or [num_classes // 2]

        def label(mu, peak):
            if len(mus) <= 1:
                return f"peak={peak}"
            elif len(peaks) <= 1:
                return f"μ={mu:g}"
            return f"μ={mu:g}, peak={peak}"

        class_frequencies_by_mu = {
            label(mu, peak): get_skellam_class_weights(mu, num_classes, peak)
            for mu, peak in itertools.product(mus, peaks)
        }
        return utils.plot_class_distributions(class_frequencies_by_mu, title="")


def generate_and_distribute_synthetic_dataset(
    globally_balanced: bool,
    locally_balanced: bool,
    num_samples: int,
    num_classes: int,
    rank: int,
    num_ranks: int,
    seed: int,
    test_size: float,
    mu_partition: Optional[Union[float, str]] = None,
    mu_data: Optional[Union[float, str]] = None,
    peak: Optional[int] = None,
    rescale_to_sum_one: bool = True,
    make_classification_kwargs: Optional[Dict] = None,
    sampling: bool = False,
    shared_test_set: bool = True,
    stratified_train_test: bool = True,
) -> Tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
    """
    Generate a synthetic dataset, partition it among all ranks and determine the subset assigned to this rank.

    Parameters
    ----------
    globally_balanced : bool
            Whether the class distribution of the entire dataset is balanced. If False, `mu_data` must be
                        specified.
    locally_balanced : bool
            Whether to use a balanced partition when assigning the dataset to ranks. If False, `mu_partition`
                       must be specified.
    num_samples : int
            The number of samples in the dataset.
    num_classes : int
            The number classes in the dataset.
    rank : int
            The index of this rank.
    num_ranks : int
            The total number of ranks.
    seed : int
            The random seed, used for both the dataset generation and the partition and distribution.
    test_size : float
            Relative size of the test set.
    mu_partition : Optional[Union[float, str]]
            The μ parameter of the skellam distribution for imbalanced class distribution. Has no effect if
            `locally_balanced` is True.
    mu_data : Optional[Union[float, str]]
            The μ parameter of the skellam distribution for imbalanced class distribution in the dataset. Has no
            effect if `globally_balanced` is True.
    peak : Optional[int]
            The position (class index) of the class distribution peak in the dataset. Has no effect if
           `globally_balanced` is True.
    rescale_to_sum_one : bool
            Whether to rescale the class weights, so they sum to 1. Default: True.
    make_classification_kwargs : Optional[Dict]
            Additional kwargs to sklearn.datasets.make_classification.
    sampling : bool
            Whether to partition the dataset using deterministic element counts and shuffling or random sampling.
            See `SyntheticDataset.get_local_subset` for more details.
    shared_test_set : bool
            Whether the test set is shared among all ranks or each rank has its own test set that is not shared with the
            other ranks. In practice, this decides whether we first split the data into train-test and then distribute
            the train set (shared test set) or first distribute the dataset and then split the local subsets into train
            and test (private test sets).
    stratified_train_test : bool
            Whether to stratify the train-test split with the class labels.

    Returns
    -------
    Tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]
            The global dataset and the local subset as X and y containing only the samples assigned to this rank.
    """
    # generate dataset
    make_classification_kwargs = (
        {} if make_classification_kwargs is None else make_classification_kwargs
    )
    if globally_balanced:
        global_dataset = SyntheticDataset.generate(
            num_samples, num_classes, None, seed, **make_classification_kwargs
        )
    else:
        global_dataset = SyntheticDataset.generate_with_skellam_class_imbalance(
            num_samples,
            num_classes,
            mu_data,
            peak,
            rescale_to_sum_one,
            seed,
            **make_classification_kwargs,
        )

    if shared_test_set:  # Case 1: shared test set, all ranks use the same test set
        # first: train test split
        global_train_set, global_test_set = global_dataset.train_test_split(
            test_size, stratified_train_test, seed
        )
        local_test = global_test_set  # in this case, the local test set is the same as the global test set
        # then: partition only the training data and distribute them across ranks
        local_train = global_train_set.get_local_subset(
            rank,
            num_ranks,
            seed,
            balanced=locally_balanced,
            mu=mu_partition,
            sampling=sampling,
        )
    else:  # Case 2: private test set, each rank has its  own test set that is not shared with the other ranks
        # first: partition the entire dataset
        local_subset = global_dataset.get_local_subset(
            rank,
            num_ranks,
            seed,
            balanced=locally_balanced,
            mu=mu_partition,
            sampling=sampling,
        )
        # then: perform train test splits locally on each rank
        local_train, local_test = local_subset.train_test_split(
            test_size, stratified_train_test, seed
        )

    return global_dataset, local_train, local_test


def data_generation_demo(
    globally_balanced: bool,
    locally_balanced: bool,
    shared_test_set: bool,
    num_samples: int = 100,
    num_classes: int = 5,
    num_ranks: int = 4,
    seed: int = 0,
    mu_partition: Union[float, str] = 2,
    mu_data: Union[float, str] = 2,
    peak: int = 0,
    test_size: float = 0.2,
    path: Optional[Union[pathlib.Path, str]] = None,
) -> None:
    print(f"\n{globally_balanced=}, {locally_balanced=}, {shared_test_set=}")
    label = f"{shared_test_set=}__{globally_balanced=}__{locally_balanced=}"
    local_datasets = {"train": [], "test": []}

    for rank in range(num_ranks):
        print(f"---- Rank {rank} ------------------------")
        (
            global_dataset,
            local_train,
            local_test,
        ) = generate_and_distribute_synthetic_dataset(
            globally_balanced=globally_balanced,
            locally_balanced=locally_balanced,
            shared_test_set=shared_test_set,
            num_samples=num_samples,
            num_classes=num_classes,
            rank=rank,
            num_ranks=num_ranks,
            seed=seed,
            mu_partition=mu_partition,
            mu_data=mu_data,
            peak=peak,
            test_size=test_size,
        )

        print(f"Global: {global_dataset}")
        print(f" Train: {local_train}")
        print(f"  Test: {local_test}\n")

        local_datasets["train"].append(local_train)
        local_datasets["test"].append(local_test)

    if path:
        local_class_frequencies = {
            key: np.array(
                [
                    [
                        dataset.get_class_frequency(True).get(class_id, 0)
                        for class_id in range(num_classes)
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


if __name__ == "__main__":
    output_path = pathlib.Path(__file__).parent.parent / "results" / "breaking_iid_demo"
    output_path.mkdir(parents=True, exist_ok=True)

    for globally_balanced, locally_balanced, shared_test_set in itertools.product(
        *[[True, False]] * 3
    ):
        data_generation_demo(
            num_classes=11,
            num_samples=10000,
            path=output_path,
            mu_partition=5,
            mu_data=5,
            peak=5,
            globally_balanced=globally_balanced,
            locally_balanced=locally_balanced,
            shared_test_set=shared_test_set,
        )

    # impact of mu
    fig, _ = SyntheticDataset.plot_skellam_distributions(
        11, mus=[0, 0.1, 0.5, 1, 2, 5, float("inf")]
    )
    fig.savefig(output_path / "skellam_distributions.svg")
