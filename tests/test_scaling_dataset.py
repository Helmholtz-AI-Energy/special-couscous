import argparse
import logging
import pathlib
from typing import cast

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from specialcouscous.scaling_dataset import (
    generate_scaling_dataset,
    read_scaling_dataset_from_hdf5,
    write_scaling_dataset_to_hdf5, generate_and_save_dataset, generate_and_save_dataset_memory_efficient,
    reproduce_random_state_before_useless_features,
)
from specialcouscous.synthetic_classification_data import SyntheticDataset
from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.

set_logger_config(
    level=logging.INFO,  # Logging level
    log_file=None,  # Logging path
    log_to_stdout=True,  # Print log on stdout.
    log_rank=False,  # Do not prepend MPI rank to logging messages.
    colors=True,  # Use colors.
)


def test_create_scaling_dataset() -> None:
    """
    Test creating a scaling dataset.

    Create a scaling dataset and ensure all splits (test set and all local train sets) have the correct size and are
    (roughly) from the same distribution by ensuring the feature- and class-wise means of all splits are within one
    standard deviation of the mean of the global training set.
    Note: Since all means seem to be close to 0 with stds around 1, it is unclear how meaningful this check really is.
    """
    # Use mostly default parameters from `utils.parse_arguments()`.
    n_classes = 4  # Use only 4 classes.
    n_features = 20  # Use only 20 features.
    frac_informative = 0.1
    frac_redundant = 0.1
    make_classification_kwargs = {
        "n_clusters_per_class": 1,
        "n_informative": int(frac_informative * n_features),
        "n_redundant": int(frac_redundant * n_features),
        "flip_y": 0,  # Don't flip labels to check number of samples per slice more easily.
    }

    # Create scaling dataset for 10 ranks.
    global_train_set, training_slices, global_test_set = generate_scaling_dataset(
        n_samples=1250,
        n_features=n_features,
        n_classes=n_classes,
        n_ranks=10,
        random_state=0,
        test_size=0.2,
        make_classification_kwargs=make_classification_kwargs,
        stratified_train_test=True,
    )
    # Just to shutup mypy: Since we don't pass a rank, we have a dict of all ranks, not just a single dataset for one.
    training_slices = cast(dict[int, SyntheticDataset], training_slices)

    # Check dataset size.
    # Global train set = 1250 * 0.8 = 1000 samples
    assert global_train_set.n_samples == 1000
    # Global test set = 1250 - 1000 = 250 samples
    assert global_test_set.n_samples == 250
    for local_train_set in training_slices.values():
        # Local train sets = 1000 / 10 (n_ranks) = 100 samples
        assert local_train_set.n_samples == 100

    # Check dataset distribution: Class- and feature-wise mean should not differ too much between the datasets,
    # e.g., mean of the first feature for class 0 should remain fairly consistent across all dataset slices.
    def subset_for_class(dataset: SyntheticDataset, class_index: int) -> np.ndarray:
        return dataset.x[dataset.y == class_index]

    for class_index in range(n_classes):
        global_train_subset = subset_for_class(global_train_set, class_index)
        global_train_mean = global_train_subset.mean(axis=0)
        global_train_std = global_train_subset.std(axis=0)

        global_test_mean = subset_for_class(global_test_set, class_index).mean(axis=0)

        print(f"\nClass {class_index}\nGlobal train mean: {global_train_mean[:5]}")
        print(f"Global test mean:  {global_test_mean[:5]}")
        print(f"Global train std:  {global_train_std[:5]}")

        # Check mean: Mean of all smaller subsets should be within std of mean from global train set.
        # Note: this check does not seem super informative since all means seem to be close to 0 with stds close to 1
        # Maybe there is another, more informative check we could make here? (see Issue #30)
        def check_feature_mean(
            mean_to_check: np.ndarray,
            baseline_mean: np.ndarray,
            allowed_difference: np.ndarray,
        ) -> bool:
            return (abs(mean_to_check - baseline_mean) < allowed_difference).all()

        assert check_feature_mean(global_test_mean, global_train_mean, global_train_std)

        for local_train_set in training_slices.values():
            local_train_mean = subset_for_class(local_train_set, class_index).mean(
                axis=0
            )
            assert check_feature_mean(
                local_train_mean, global_train_mean, global_train_std
            )


def test_write_read_scaling_dataset(tmp_path: pathlib.Path) -> None:
    """
    Test the writing and reading of scaling datasets.

    Generate a scaling dataset and test
    - writing it to HDF5.
    - writing it to HDF5 when the matching file already exists (once with, once without override).
    - reading it from HDF5 and comparing it to the original (both reading all ranks at once and rank by rank).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path to write the data to.
    """
    # Use mostly default parameters from `utils.parse_arguments()`.
    n_classes = 10
    n_features = 100
    frac_informative = 0.1
    frac_redundant = 0.1
    make_classification_kwargs = {
        "n_clusters_per_class": 1,
        "n_informative": int(frac_informative * n_features),
        "n_redundant": int(frac_redundant * n_features),
        "flip_y": 0.01,
    }
    n_ranks = 10

    # Create scaling dataset for 10 ranks.
    global_train_set, training_slices, global_test_set = generate_scaling_dataset(
        n_samples=1250,
        n_features=n_features,
        n_classes=n_classes,
        n_ranks=n_ranks,
        random_state=0,
        test_size=0.2,
        make_classification_kwargs=make_classification_kwargs,
        stratified_train_test=True,
    )
    # Just to shutup mypy: Since we don't pass a rank, we have a dict of all ranks, not just a single dataset for one.
    training_slices = cast(dict[int, SyntheticDataset], training_slices)

    hdf5_path = tmp_path / "dataset.h5"
    global_attrs = {"test": "test_attr"}

    # Write scaling dataset to HDF5.
    write_scaling_dataset_to_hdf5(
        global_train_set, training_slices, global_test_set, global_attrs, hdf5_path
    )

    # Check overriding: Raises exception if `override` is False but overrides if True.
    pytest.raises(
        FileExistsError,
        lambda: write_scaling_dataset_to_hdf5(
            global_train_set,
            training_slices,
            global_test_set,
            global_attrs,
            hdf5_path,
            override=False,
        ),
    )
    write_scaling_dataset_to_hdf5(
        global_train_set,
        training_slices,
        global_test_set,
        global_attrs,
        hdf5_path,
        override=True,
    )

    expected_root_attrs = {
        **global_attrs,
        "n_classes": n_classes,
        "n_ranks": n_ranks,
        "n_samples_global_train": global_train_set.n_samples,
    }

    # Read entire scaling dataset from HDF5.
    read_local_train_sets, read_global_test_set, read_root_attrs = (
        read_scaling_dataset_from_hdf5(hdf5_path)
    )
    assert read_global_test_set == global_test_set
    assert read_local_train_sets == training_slices
    assert read_root_attrs == expected_root_attrs

    # Read specific local training set from HDF5.
    for rank in range(n_ranks):
        read_local_train_set, read_global_test_set, read_root_attrs = (
            read_scaling_dataset_from_hdf5(hdf5_path, rank=rank)
        )
        assert read_global_test_set == global_test_set
        assert read_local_train_set == training_slices[rank]
        assert read_root_attrs == expected_root_attrs


@pytest.fixture()
def default_args(tmp_path: pathlib.Path) -> argparse.Namespace:
    """
    Initialize a default namespace as could be parsed by parse_arguments().

    Returns
    -------
    argparse.Namespace
        The namespace filled with default values.
    tmp_path : pathlib.Path
        Temporary path to write the data to (set as data_root_path).
    """
    args = argparse.Namespace()

    args.n_samples = 100
    args.n_features = 20
    args.n_classes = 2
    args.n_train_splits = 1
    args.random_state = 0
    args.train_split = 0.2
    args.stratified_train_test = True
    args.n_clusters_per_class = 1
    args.frac_informative = 0.1
    args.frac_redundant = 0.1
    args.flip_y = 0.01
    args.data_root_path = tmp_path
    args.override_data = False

    return args


def test_reproduce_random_state_before_useless_features() -> None:
    """
    Test the reproduction of the random state during make_classification.

    The random state after make_classification with shuffle=False, flip_y=0.0 and no useless features and specified
    shift and scale should be the same as the one generated by reproduce_random_state_before_useless_features (based on
    the same random seed) followed by generating the flip mask (since the mask is always generated, even for flip == 0).
    """

    def random_state_eq(first: np.random.RandomState, second: np.random.RandomState) -> bool:
        """
        Ensure that two numpy random states are the same by comparing their state.

        Parameters
        ----------
        first : np.random.RandomState
            The first random state.
        second : np.random.RandomState
            The second random state.

        Returns
        -------
        bool
            True if the two are the same.

        Raises
        -------
        AssertionError
            If the two are different (based on their state).
        """
        first_state = first.get_state()
        second_state = second.get_state()

        # expect tuples of len 5, see numpy.random.get_state()
        assert len(first_state) == 5
        assert len(second_state) == 5

        assert first_state[0] == second_state[0]
        assert first_state[2] == second_state[2]
        assert first_state[3] == second_state[3]
        assert first_state[4] == second_state[4]

        assert np.array_equal(first_state[1], second_state[1])

        return True

    random_seed = 0
    random_state_baseline = check_random_state(random_seed)
    random_state_reproduced = check_random_state(random_seed)
    random_state_eq(random_state_baseline, random_state_reproduced)

    n_samples = 100
    make_classification_kwargs = {"n_samples": n_samples, "n_classes": 10, "n_clusters_per_class": 2,
                                  "n_informative": 10, "n_redundant": 5, "n_repeated": 2}
    n_features = sum(v for k, v in make_classification_kwargs.items()
                     if k in ["n_informative", "n_redundant", "n_repeated"])

    make_classification(random_state=random_state_baseline, n_features=n_features, shuffle=False, flip_y=0.0,
                        **make_classification_kwargs)

    random_state_returned = reproduce_random_state_before_useless_features(
        random_state_reproduced, **make_classification_kwargs)
    assert random_state_returned is random_state_reproduced
    random_state_reproduced.uniform(size=n_samples)

    random_state_eq(random_state_reproduced, random_state_baseline)

    for _ in range(10):
        assert random_state_reproduced.randint(100) == random_state_baseline.randint(100)


@pytest.mark.parametrize("frac_useless", [0.0, 0.2, 0.8])
def test_memory_efficient_scaling_dataset(default_args, tmp_path: pathlib.Path, frac_useless: float) -> None:
    """
    Test the memory efficient dataset generation

    Generate a scaling dataset and test
    - writing it to HDF5.
    - writing it to HDF5 when the matching file already exists (once with, once without override).
    - reading it from HDF5 and comparing it to the original (both reading all ranks at once and rank by rank).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path to write the data to.
    frac_useless : float
        Fraction of useless features.
    """
    #
    default_args.flip_y = 0.0
    default_args.frac_informative = (1. - frac_useless) / 2.
    default_args.frac_redundant = (1. - frac_useless) / 2.
    log.info(f'frac_informative: {default_args.frac_informative}, frac_redundant: {default_args.frac_redundant}')
    shuffle = False
    file_name = ("n_samples_{n_samples}__n_features_{n_features}__n_classes_{n_classes}/"
                 "{n_train_splits}_ranks__seed_{random_state}.h5").format(**vars(default_args))

    # Generate two datasets: one with the original approach, one with the memory efficient approach.
    default_args.data_root_path = tmp_path / "original"
    generate_and_save_dataset(default_args, shuffle)
    default_args.data_root_path = tmp_path / "memory_efficient"
    generate_and_save_dataset_memory_efficient(default_args, shuffle, reproduce_random_state=True)

    # Read both datasets back from the HDF5 file.
    original_local_train_sets, original_global_test_set, original_root_attrs = (
        read_scaling_dataset_from_hdf5(tmp_path / "original" / file_name)
    )
    memory_efficient_local_train_sets, memory_efficient_global_test_set, memory_efficient_root_attrs = (
        read_scaling_dataset_from_hdf5(tmp_path / "memory_efficient" / file_name)
    )

    # Check if both HDF5 files store the same datasets (both global test and all local train sets)
    assert original_local_train_sets == memory_efficient_local_train_sets
    assert original_global_test_set == memory_efficient_global_test_set

    # Check if the attributes are the same except for the "memory_efficient_generation" value
    assert not original_root_attrs["memory_efficient_generation"]
    assert memory_efficient_root_attrs["memory_efficient_generation"]

    def other_attributes(attributes):
        return {k: v for k, v in attributes.items() if k != "memory_efficient_generation"}

    assert other_attributes(original_root_attrs) == other_attributes(memory_efficient_root_attrs)
