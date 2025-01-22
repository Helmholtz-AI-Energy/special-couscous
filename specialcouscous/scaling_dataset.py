from __future__ import annotations

import argparse
import logging
import os
import pathlib
from typing import Any, cast

import h5py
import numpy as np
from sklearn.utils import check_random_state

from specialcouscous.synthetic_classification_data import (
    DatasetPartition,
    SyntheticDataset,
)
from specialcouscous.utils import parse_arguments, set_logger_config

# Get logger: specialcouscous.<filename>
__FILE_PATH = pathlib.Path(__file__)
log = logging.getLogger(f"{__FILE_PATH.parent.name}.{__FILE_PATH.stem}")


def generate_scaling_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_ranks: int,
    test_size: float,
    random_state: int | np.random.RandomState,
    random_state_slicing: int | np.random.RandomState | None = None,
    make_classification_kwargs: dict[str, Any] | None = None,
    sampling: bool = False,
    stratified_train_test: bool = False,
    rank: int | None = None,
) -> tuple[
    SyntheticDataset, dict[int, SyntheticDataset] | SyntheticDataset, SyntheticDataset
]:
    """
    Generate a dataset to be scaled with the number of nodes.

    Generates a single global dataset and splits it into ``n_ranks`` slices. Returns the global train and test set and
    either the local train set of the given rank, or a dict of all local train sets (mapping rank -> local train set)
    when rank is None.

    Parameters
    ----------
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number of classes in the dataset.
    n_ranks : int
        The total/maximum number of ranks to generate the dataset for.
    test_size : float
        Relative size of the test set.
    random_state : int | np.random.RandomState
        The random state, used for dataset generation. If no random_state_slicing is passed, this random state is also
        used for partition and distribution.
    random_state_slicing : int | np.random.RandomState | None
        The random state used for dataset partition and distribution. If None, the general random_state will be used.
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
    random_state_slicing = random_state if random_state_slicing is None else check_random_state(random_state_slicing)

    log.debug(f"Classification kwargs: {make_classification_kwargs}")

    # Step 1: Generate balanced global dataset for up to `n_ranks` nodes.
    global_dataset = SyntheticDataset.generate(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
        make_classification_kwargs=make_classification_kwargs,
    )

    # Step 2: Split into global train and test set.
    # TODO: in this case, the size of the test set scales with n_ranks -> do we want this? if not, what else?
    log.debug(
        f"Generate global train-test split: {test_size=}, {stratified_train_test=}."
    )
    global_train_set, global_test_set = global_dataset.train_test_split(
        test_size=test_size,
        stratify=stratified_train_test,
        random_state=random_state_slicing,
    )
    log.debug(
        f"Shape of global train set {global_train_set.x.shape}, test set {global_test_set.x.shape}"
    )

    # Step 3: Partition the global train set into `n_ranks` local train sets (balanced partition).
    partition = DatasetPartition(global_train_set.y)
    assigned_ranks = partition.balanced_partition(n_ranks, random_state_slicing, sampling)
    assigned_indices = partition.assigned_indices_by_rank(assigned_ranks)
    training_slices = {
        rank: SyntheticDataset(
            global_train_set.x[indices], global_train_set.y[indices], None, n_classes
        )
        for rank, indices in assigned_indices.items()
    }

    log.debug(f"Shape of local train set 0 is {training_slices[0].x.shape}.")

    if rank is not None:
        log.debug(f"Returning local train set for rank {rank}")
        return global_train_set, training_slices[rank], global_test_set
    else:
        log.debug("Returning dict of all local train sets")
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
    Write a scaling dataset as HDF5 file to the given path.

    The HDF5 file has the following structure:
    Root Attributes:
    - '/n_classes': the number of classes in the dataset
    - '/n_ranks': the number of ranks into which the dataset has been partitioned
    - '/n_samples_global_train': the number of samples in the global training set
    - **additional_global_attrs: additional_global_attrs are written directly to the root attributes
      can for example be used to store the config used to generate this dataset

    Groups:
    - '/test_set': the global test_set
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
        A dict of all local train sets by rank.
    global_test_set : SyntheticDataset
        The global = local test set.
    additional_global_attrs : dict[str, Any]
        A dict of additional attributes to write root attributes to the HDF5 file. Can for example be used to save the
        config parameters used to generate this dataset.
    file_path : os.PathLike
        The file path of the HDF5 file to write.
    override : bool
        If True, will override any existing files at ``file_path``. If False, will raise a ``FileExistsError``
        should ``file_path`` already exist.
    """
    file_path = pathlib.Path(file_path)
    if not override and file_path.exists():
        raise FileExistsError(
            f"File {file_path} exists and override is set to {override}."
        )
    file = h5py.File(file_path, "w")

    file.attrs["n_classes"] = global_train_set.n_classes
    file.attrs["n_ranks"] = len(local_train_sets)
    file.attrs["n_samples_global_train"] = global_train_set.n_samples
    for key, value in additional_global_attrs.items():
        log.debug(f"Adding attr {key}={value}")
        file.attrs[key] = value

    def write_subset_to_group(
        group_name: Any, dataset: SyntheticDataset, **attrs: Any
    ) -> None:
        file[f"{group_name}/x"] = dataset.x
        file[f"{group_name}/y"] = dataset.y
        file[group_name].attrs["n_samples"] = dataset.n_samples
        for key, value in attrs.items():
            file[group_name].attrs[key] = value

    # Write global test set to HDF5.
    write_subset_to_group("test_set", global_test_set, label="global_test_set")

    # Write local train sets to HDF5.
    for rank, local_train_set in local_train_sets.items():
        group_name = f"local_train_sets/rank_{rank}"
        write_subset_to_group(group_name, local_train_set, label=group_name, rank=rank)
    log.info(f"Dataset successfully written to {file_path}.")
    log.info(
        "To use this dataset, call `scaling_dataset.load_and_verify_dataset(args)` with the same CLI arguments."
    )


def read_scaling_dataset_from_hdf5(
    file_path: os.PathLike,
    rank: int | None = None,
) -> tuple[
    dict[int, SyntheticDataset] | SyntheticDataset, SyntheticDataset, dict[str, Any]
]:
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

    def dataset_from_group(group: h5py.Group) -> SyntheticDataset:
        return SyntheticDataset(
            group["x"],
            group["y"],
            n_samples=int(group.attrs["n_samples"]),
            n_classes=int(n_classes),
        )

    global_test_set = dataset_from_group(file["test_set"])

    local_train_set: (
        dict[int, SyntheticDataset] | SyntheticDataset
    )  # Define type first for mypy.
    if rank is None:  # Dict of all local train sets
        local_train_set = {
            group.attrs["rank"]: dataset_from_group(group)
            for name, group in file["local_train_sets"].items()
        }
    else:  # Just the local train set for the given rank
        local_train_set = dataset_from_group(file[f"local_train_sets/rank_{rank}"])

    return local_train_set, global_test_set, root_attrs


def dataset_path(
    root_path: os.PathLike,
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_nodes: int,
    seed: int,
) -> pathlib.Path:
    """
    Construct the dataset path depending on its parameters and make sure all parent directories exist.

    Parameters
    ----------
    root_path : os.PathLike
        The path to the root directory for all datasets.
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_classes : int
        The number of classes.
    n_nodes : int
        The number of nodes = number of splits in the training set.
    seed : int
        The seed used for the random state of the data generation.

    Returns
    -------
    pathlib.Path
        The path to the HDF5 file for the corresponding dataset.
    """
    root_path = pathlib.Path(root_path)
    # n = log10 number of samples, m = log10 number of features
    path = (
        root_path
        / f"n_samples_{n_samples}__n_features_{n_features}__n_classes_{n_classes}"
        / f"{n_nodes}_ranks__seed_{seed}.h5"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path_from_args(args: argparse.Namespace) -> pathlib.Path:
    """
    Generate ``dataset_path`` directly from args object from the CLI parser (wrapper for ``dataset_path``).

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.

    Returns
    -------
    pathlib.Path
        The path to the HDF5 file for the corresponding dataset.
    """
    if args.n_train_splits is None:
        raise ValueError(
            "n_train_splits is required for pre-generated datasets. Please specify --n_train_splits."
        )

    return dataset_path(
        root_path=args.data_root_path,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_nodes=args.n_train_splits,
        seed=args.random_state,
    )


def dataset_config_from_args(
    args: argparse.Namespace, unpack_kwargs: bool = False
) -> dict[str, Any]:
    """
    Convert the CLI parameters to the configuration passed to ``generate_scaling_dataset``.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.
    unpack_kwargs : bool
        Whether to unpack the make_classification_kwargs or keep them as nested dict.
        Leave as False to get correct config for dataset generation, set to True to unpack the nested dict, e.g., for
        using the config as HDF5 attributes.

    Returns
    -------
    dict[str, Any]
        The configuration parameters as dict.
    """
    if args.n_train_splits is None:
        raise ValueError(
            "n_train_splits is required for pre-generated datasets. Please specify --n_train_splits."
        )
    general_kwargs = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "n_classes": args.n_classes,
        "n_ranks": args.n_train_splits,
        "random_state": args.random_state,
        "test_size": 1 - args.train_split,
        "stratified_train_test": args.stratified_train_test,
    }
    make_classification_kwargs = {
        "n_clusters_per_class": args.n_clusters_per_class,
        "n_informative": int(args.frac_informative * args.n_features),
        "n_redundant": int(args.frac_redundant * args.n_features),
        "flip_y": args.flip_y,
    }
    if unpack_kwargs:
        return {**general_kwargs, **make_classification_kwargs}
    else:
        return {
            **general_kwargs,
            "make_classification_kwargs": make_classification_kwargs,
        }


def load_and_verify_dataset(
    args: argparse.Namespace, rank: int | None, fail_on_unmatched_config: bool = False
) -> tuple[
    dict[int, SyntheticDataset] | SyntheticDataset, SyntheticDataset, dict[str, Any]
]:
    """
    Load dataset from HDF5 and verify that the configuration matches the CLI arguments.

    Try to read the corresponding dataset from HDF5 and verify that the actual dataset config stored in the HDF5 matches
    the config inferred from the current CLI parameter.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.
    rank : int | None
        The rank whose local train set shall be loaded. Set to None to load all ranks.
    fail_on_unmatched_config : bool
        If true, an error is raised if the verification fails. Otherwise, only an error message is printed and the
        (potentially mismatched) dataset is returned anyway.

    Returns
    -------
    dict[int, SyntheticDataset] | SyntheticDataset
        A dict of all local train sets by rank or just the local train set for the given rank.
    SyntheticDataset
        The global = local test set.
    dict[str, Any]
        Any additional information on the dataset stored as root attribute in the HDF5 file.
    """
    # Read dataset from HDF5.
    path = dataset_path_from_args(args)
    local_train_sets, global_test_set, attrs = read_scaling_dataset_from_hdf5(
        path, rank=rank
    )

    # Verify that the metadata stored within the HDF5 is identical to that specified by the parameters.
    expected_dataset_config = dataset_config_from_args(args)
    expected_dataset_config = {
        **expected_dataset_config,
        **expected_dataset_config["make_classification_kwargs"],
    }
    del expected_dataset_config["make_classification_kwargs"]
    actual_dataset_config = {
        key: value for key, value in attrs.items() if key in expected_dataset_config
    }

    if expected_dataset_config != actual_dataset_config:
        error_message = (
            f"Dataset config does not match current CLI arguments. "
            f"From CLI {expected_dataset_config}, actual in HDF5 {actual_dataset_config}."
        )
        if fail_on_unmatched_config:
            raise ValueError(error_message)
        log.warning(f"Warning: {error_message}")

    return local_train_sets, global_test_set, attrs


def generate_and_save_dataset(args: argparse.Namespace) -> None:
    """
    Generate a scaling dataset based on the given CLI parameters.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.
    """
    # Generate the dataset.
    args.random_state_slicing = args.random_state
    dataset_config = dataset_config_from_args(args, unpack_kwargs=False)
    log.info(f"Creating dataset with the following parameters:\n{dataset_config}")
    global_train_set, local_train_sets, global_test_set = generate_scaling_dataset(
        **dataset_config
    )
    # Just to shutup mypy: Since we don't pass a rank, we have a dict of all ranks, not just a single dataset for one.
    local_train_sets = cast(dict[int, SyntheticDataset], local_train_sets)

    # Write the dataset to HDF5.
    path = dataset_path_from_args(args)
    attrs = dataset_config_from_args(args, unpack_kwargs=True)
    write_scaling_dataset_to_hdf5(
        global_train_set,
        local_train_sets,
        global_test_set,
        attrs,
        path,
        args.override_data,
    )


if __name__ == "__main__":
    set_logger_config(level=logging.DEBUG)
    args = parse_arguments()
    generate_and_save_dataset(args)
