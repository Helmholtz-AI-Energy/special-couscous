import logging
import pathlib
from typing import cast

import numpy as np
import pytest

from specialcouscous.scaling_dataset import (
    generate_scaling_dataset,
    read_scaling_dataset_from_hdf5,
    write_scaling_dataset_to_hdf5,
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
        # TODO: this check does not seem super informative since all means seem to be close to 0 with stds close to 1
        # Maybe there is another, more informative check we could make here?
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
