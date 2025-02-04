from __future__ import annotations

import argparse
import logging
import os
import pathlib
from typing import Any, cast

import h5py
import numpy as np
from sklearn.datasets._samples_generator import _generate_hypercube
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
    make_classification_kwargs = (
        {} if make_classification_kwargs is None else make_classification_kwargs
    )
    random_state = check_random_state(random_state)
    random_state_slicing = (
        random_state
        if random_state_slicing is None
        else check_random_state(random_state_slicing)
    )

    log.debug(f"Classification kwargs: {make_classification_kwargs}")

    # Step 1: Generate balanced global dataset for up to `n_ranks` nodes.
    global_dataset = SyntheticDataset.generate(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
        make_classification_kwargs=make_classification_kwargs,
    )
    log.debug(
        f"Pos of random_state_generation after generate: {random_state.get_state()[2]}"
    )

    # Step 2: Split into global train and test set.
    kwargs = {"stratify": stratified_train_test}
    if not make_classification_kwargs.get("shuffle", True):
        log.debug(
            f"Shuffle is False -> global train-test split without shuffling, ignoring {stratified_train_test=}."
        )
        kwargs = {"shuffle": False, "stratify": False}
    log.debug(f"Generate global train-test split: {test_size=}.")
    global_train_set, global_test_set = global_dataset.train_test_split(
        test_size=test_size, random_state=random_state_slicing, **kwargs
    )
    log.debug(
        f"Shape of global train set {global_train_set.x.shape}, test set {global_test_set.x.shape}"
    )

    # Step 3: Partition the global train set into `n_ranks` local train sets (balanced partition).
    partition = DatasetPartition(global_train_set.y)
    assigned_ranks = partition.balanced_partition(
        n_ranks, random_state_slicing, sampling
    )
    assigned_indices = partition.assigned_indices_by_rank(assigned_ranks)
    training_slices = {
        rank: SyntheticDataset(
            global_train_set.x[indices], global_train_set.y[indices], None, n_classes
        )
        for rank, indices in assigned_indices.items()
    }
    log.debug(
        f"Current pos of random_state_slicing: {random_state_slicing.get_state()[2]}"
    )

    for i in range(n_ranks):
        log.debug(f"Shape of local train set {i: 2d} is {training_slices[i].x.shape}.")

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
    with_global_test: bool = True,
) -> tuple[
    dict[int, SyntheticDataset] | SyntheticDataset,
    SyntheticDataset | None,
    dict[str, Any],
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
    with_global_test : bool
        Whether to include the global testset or skip reading it and return None as second return value instead.

    Returns
    -------
    dict[int, SyntheticDataset] | SyntheticDataset
        A dict of all local train sets by rank or just the local train set for the given rank.
    SyntheticDataset | None
        The global = local test set if with_global_test is True, None otherwise.
    dict[str, Any]
        Any additional information on the dataset stored as root attribute in the HDF5 file.
    """
    file = h5py.File(file_path, "r")
    n_classes = file.attrs["n_classes"]

    def unpack_numpy_types(value: Any) -> Any:
        """
        Try unpacking numpy types to standard python types (e.g. np.bool_ to bool).

        Parameters
        ----------
        value : Any
            The value to try unpacking.

        Returns
        -------
        Either value.item() if value is a numpy type or the unchanged value.
        """
        is_numpy_type = type(value).__module__ == np.__name__
        return value.item() if is_numpy_type else value

    root_attrs = {key: unpack_numpy_types(value) for key, value in file.attrs.items()}

    def dataset_from_group(group: h5py.Group) -> SyntheticDataset:
        return SyntheticDataset(
            group["x"][...],
            group["y"][...],
            n_samples=int(group.attrs["n_samples"]),
            n_classes=int(n_classes),
        )

    global_test_set = dataset_from_group(file["test_set"]) if with_global_test else None

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
    args: argparse.Namespace, unpack_kwargs: bool = False, shuffle: bool | None = None
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
    shuffle : bool | None
        Optional way to set the shuffle parameter in make_classification_kwargs,

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
        "random_state_slicing": args.random_state_slicing,
        "test_size": 1 - args.train_split,
        "stratified_train_test": args.stratified_train_test,
    }
    make_classification_kwargs = {
        "n_clusters_per_class": args.n_clusters_per_class,
        "n_informative": int(args.frac_informative * args.n_features),
        "n_redundant": int(args.frac_redundant * args.n_features),
        "flip_y": args.flip_y,
    }
    if shuffle is not None:
        make_classification_kwargs["shuffle"] = shuffle
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
    global_test_set = cast(SyntheticDataset, global_test_set)

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


def generate_and_save_dataset(
    args: argparse.Namespace, shuffle: bool | None = None
) -> None:
    """
    Generate a scaling dataset based on the given CLI parameters.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.
    shuffle : bool | None
        The shuffle parameter for make_classification. Set this to False, together with setting flip_y < 0 to obtain
        identical results with the memory efficient data generation approach.
    """
    # Generate the dataset.
    args.random_state_slicing = args.random_state
    dataset_config = dataset_config_from_args(
        args, unpack_kwargs=False, shuffle=shuffle
    )
    log.info(f"Creating dataset with the following parameters:\n{dataset_config}")
    global_train_set, local_train_sets, global_test_set = generate_scaling_dataset(
        **dataset_config
    )
    # Just to shutup mypy: Since we don't pass a rank, we have a dict of all ranks, not just a single dataset for one.
    local_train_sets = cast(dict[int, SyntheticDataset], local_train_sets)

    # Write the dataset to HDF5.
    path = dataset_path_from_args(args)
    attrs = dataset_config_from_args(args, unpack_kwargs=True, shuffle=shuffle)
    attrs["memory_efficient_generation"] = False
    write_scaling_dataset_to_hdf5(
        global_train_set,
        local_train_sets,
        global_test_set,
        attrs,
        path,
        args.override_data,
    )


def add_useless_features(
    x: np.ndarray,
    n_useless: int,
    random_state: np.random.RandomState,
    shuffle: bool,
    shift: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Add useless noise features to a given array of features.

    Generate n_useless additional features by sampling random noise from a standard normal distribution, shifting and
    scaling by shift/scale. If shuffle is True, the features (but not the samples) are shuffled.

    Parameters
    ----------
    x : np.ndarray
        The useful features.
    n_useless : int
        The number of useless features to add.
    random_state : np.random.RandomState
        The random state to use for sampling the features.
    shuffle : bool
        Whether to shuffle the features after appending the new features.
    shift : float
        The value by which to shift the random features (before scaling).
    scale : float
        The value by which to scale the random features (after shifting).

    Returns
    -------
    np.ndarray
        The feature array x with the new, useless features added.
    """
    # Create useless features from random noise
    n_samples = x.shape[0]
    useless_features = random_state.standard_normal(size=(n_samples, n_useless))
    # Shift and scale the new features
    useless_features = (useless_features + shift) * scale

    # append useless features to dataset (after existing features)
    x = np.concat([x, useless_features], axis=1)

    if shuffle:  # Shuffle features only
        feature_indices = np.arange(x.shape[1])
        random_state.shuffle(feature_indices)
        x[:, :] = x[:, feature_indices]

    return x


def add_useless_features_to_hdf5(file: h5py.File, group_name: str, random_state: np.random.RandomState, n_useless: int,
                                 shuffle: bool) -> None:
    """
    Add useless noise features a group in the given HDF5 file by replacing the group's features x.

    Parameters
    ----------
    file : h5py.File
        The HDF5 file to add useless features to.
    group_name : str
        Name of the HDF5 group to add useless features to.
    random_state : np.random.RandomState
        The random state to use for sampling the features.
    n_useless : int
        The number of useless features to add.
    shuffle : bool
        Whether to shuffle the features after appending the new features.
    """
    log.debug(
        f"Adding useless features to {group_name}."
        f"Current random state pos: {random_state.get_state()[2]}"
    )
    group = file[group_name]
    full_features = add_useless_features(group["x"], n_useless, random_state, shuffle)
    log.debug(f"Done generating useless features for {group_name}. Updating HDF5.")
    del group["x"]  # need to delete old features since we are changing the shape
    group["x"] = full_features


def reproduce_random_state_add_useless_features(
    n_samples: int,
    n_useful: int,
    n_useless: int,
    random_state: np.random.RandomState,
    shuffle: bool,
) -> np.random.RandomState:
    """
    Reproduce the random state change from add_useless_features.

    Performs only the random sampling operations of add_useless_features.

    Parameters
    ----------
    n_samples : int
        The number of samples (i.e. x.shape[0] for features x).
    n_useful : int
        The number of useful features (information, redundant, repeated) (i.e. x.shape[1] for useful features x).
    n_useless : int
        The number of useless features to add.
    random_state : np.random.RandomState
        The random state to use for sampling the features.
    shuffle : bool
        Whether to shuffle the features after appending the new features.

    Returns
    -------
    np.random.RandomState
        The feature array x with the new, useless features added.
    """
    random_state.standard_normal(size=(n_samples, n_useless))
    if shuffle:
        feature_indices = np.arange(n_useful + n_useless)
        random_state.shuffle(feature_indices)
    return random_state


def reproduce_random_state_before_useless_features(
    random_state: int | np.random.RandomState,
    n_samples: int,
    n_classes: int,
    n_clusters_per_class: int,
    n_informative: int,
    n_redundant: int,
    n_repeated: int = 0,
    hypercube: bool = True,
) -> np.random.RandomState:
    """
    Reproduce the random state during make_classification before the useless feature generation.

    Since make_classification will generate a flip mask even for flip_y == 0, we cannot disable all changes to the
    random seed after generating the useful features (even with shuffle=False, flip_y=0.0 and no useless features and
    specified shift and scale). To reproduce the exact same useless features, we thus need to reproduce the random state
    make_classification would have used when generating the useless features.
    This function takes a random state (or seed) and performs the same number of random generations as
    make_classification would before generating the useless features using the given parameters.

    Parameters
    ----------
    random_state : int | np.random.RandomState
        Random state or seed, represents the state before make_classification. When passing a random state, the state is
        modified and is identical to the one returned by this function.
    n_samples : int
        Number of samples passed to make_classification.
    n_classes : int
        Number of samples passed to make_classification.
    n_clusters_per_class : int
        Number of classes passed to make_classification.
    n_informative : int
        Number of informative features passed to make_classification.
    n_redundant : int
        Number of redundant features passed to make_classification.
    n_repeated : int
        Number of repeated features passed to make_classification.
    hypercube : bool
        The hypercube parameter passed to make_classification.

    Returns
    -------
    np.random.RandomState
        The random state as it would be during make_classification before generating the random features.
    """
    random_state = check_random_state(random_state)
    log.debug(f"Before pos of random_state: {random_state.get_state()[2]}")

    n_clusters = n_classes * n_clusters_per_class
    _generate_hypercube(n_clusters, n_informative, random_state)
    if not hypercube:
        random_state.standard_normal(size=(n_clusters + n_informative))

    random_state.standard_normal(
        size=(n_samples, n_informative)
    )  # informative feature creation

    uniform_sizes = [
        n_informative * n_informative * n_clusters,  # cluster creation
        n_informative * n_redundant,  # redundant feature creation
        n_repeated,  # repeated feature creation
    ]
    random_state.uniform(size=sum(uniform_sizes))
    log.debug(f"After pos of random_state: {random_state.get_state()[2]}")
    return random_state


def generate_and_save_dataset_memory_efficient(
    args: argparse.Namespace,
    shuffle: bool = True,
    reproduce_random_state: bool = False,
) -> None:
    """
    Generate a scaling dataset based on the given CLI parameters in a more memory-efficient way.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.
    shuffle : bool
        The shuffle parameter for make_classification. Set this to False to reproduce the results of the normal data
        generation approach more closely.
    reproduce_random_state : bool
        Whether to reproduce the random state exactly before generating the useless features. This can be used to
        generate the exact same dataset as make_classification (for example, for the test cases).
        Note that both shuffling, and random partition over more than one rank reorder the samples and are not
        reproduced, thus leading to slightly different results.
    """
    # Step 0: Prepare data generation config
    args.random_state_slicing = args.random_state
    dataset_config = dataset_config_from_args(
        args, unpack_kwargs=False, shuffle=shuffle
    )
    log.info(
        f"Aiming to create dataset with the following parameters:\n{dataset_config}"
    )

    # Count useful and useless features, generate only useful features for now
    n_features = args.n_features
    n_useful = sum(
        dataset_config["make_classification_kwargs"].get(key, 0)
        for key in ["n_informative", "n_redundant", "n_repeated"]
    )
    n_useless = n_features - n_useful
    dataset_config["n_features"] = n_useful
    log.info(
        f"Creating dataset with only useful features using the following parameters:\n{dataset_config}"
    )

    # Step 1: Generate global dataset without the useless features
    log.info("Start generation of global dataset without useless features.")
    global_train_set, local_train_sets, global_test_set = generate_scaling_dataset(
        **dataset_config
    )
    # Just to shutup mypy: Since we don't pass a rank, we have a dict of all ranks, not just a single dataset for one.
    local_train_sets = cast(dict[int, SyntheticDataset], local_train_sets)
    log.info("Dataset generation done.")

    # Step 2: Write the dataset without the useless features to HDF5.
    log.info("Start writing global dataset to HDF5.")
    path = dataset_path_from_args(args)
    attrs = dataset_config_from_args(args, unpack_kwargs=True, shuffle=shuffle)
    attrs["memory_efficient_generation"] = True
    write_scaling_dataset_to_hdf5(
        global_train_set,
        local_train_sets,
        global_test_set,
        attrs,
        path,
        args.override_data,
    )
    log.info(f"Done writing global dataset to HDF5 {path}.")

    # Step 3: Add useless features one-by-one to each dataset slice
    log.info("Start adding useless features to each slice.")
    log.info("Preparing random state.")
    random_state_generation = check_random_state(args.random_state)
    if reproduce_random_state:
        if shuffle or args.n_train_splits > 1:
            log.warning(
                f"Passed {reproduce_random_state=} but {shuffle=} and {args.n_train_splits=} > 1."
                "Note that shuffling and random partitioning across nodes is currently not reproduced, "
                "i.e. the values will be the same but shuffled differently."
            )
        reproduce_random_state_before_useless_features(
            random_state=random_state_generation,
            n_samples=args.n_samples,
            n_classes=args.n_classes,
            n_clusters_per_class=args.n_clusters_per_class,
            n_informative=dataset_config["make_classification_kwargs"]["n_informative"],
            n_redundant=dataset_config["make_classification_kwargs"]["n_redundant"],
        )
    log.debug(
        f"Current pos of random_state_generation: {random_state_generation.get_state()[2]}"
    )
    file = h5py.File(path, "r+")
    for group_name in [
        f"local_train_sets/{name}" for name in file["local_train_sets"]
    ] + ["test_set"]:
        add_useless_features_to_hdf5(
            file, group_name, random_state_generation, n_useless, shuffle
        )
    log.info("Done adding useless features.")


def continue_memory_efficient_dataset_generation(
    args: argparse.Namespace,
    shuffle: bool = True,
    reproduce_random_state: bool = False,
) -> None:
    """
    Continue the generation of useless features from a previous run.

    Continue a memory-efficient dataset generation run that was aborted during the useless feature generation phase.
    There needs to be a matching HDF5 file with the same parameters. The useful features are assumed to be generated
    correctly. For each slice (global test and local train sets) the HDF5 either contains only the useful features
    (then, useless features will be added) or already contain all features (useful and useless). The cases are
    differentiated by the shape of their feature array.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.
    shuffle : bool
        The shuffle parameter for make_classification. Set this to False to reproduce the results of the normal data
        generation approach more closely.
    reproduce_random_state : bool
        Whether to reproduce the random state exactly before generating the useless features. This can be used to
        generate the exact same dataset as make_classification (for example, for the test cases).
        Note that both shuffling, and random partition over more than one rank reorder the samples and are not
        reproduced, thus leading to slightly different results.
    """
    args.random_state_slicing = args.random_state
    dataset_config = dataset_config_from_args(
        args, unpack_kwargs=False, shuffle=shuffle
    )
    n_features = args.n_features
    n_useful = sum(
        dataset_config["make_classification_kwargs"].get(key, 0)
        for key in ["n_informative", "n_redundant", "n_repeated"]
    )
    n_useless = n_features - n_useful

    # Read HDF5 file and confirm matching attributes
    path = dataset_path_from_args(args)
    log.info(f"Reading HDF5 file from {path}.")
    file = h5py.File(path, "r+")
    expected_attrs = dataset_config_from_args(args, unpack_kwargs=True, shuffle=shuffle)
    expected_attrs["memory_efficient_generation"] = True
    file_attributes = {
        key: value.item()
        for key, value in file.attrs.items()
        if key != "n_samples_global_train"
    }
    if file_attributes != expected_attrs:
        print(file_attributes.keys() - expected_attrs.keys())
        print(expected_attrs.keys() - file_attributes.keys())
        raise ValueError(
            f"Mismatch dataset attributes: expected {expected_attrs} but got {file_attributes} from HDF5."
        )

    # Prepare random state
    log.info("Preparing random state.")
    random_state = check_random_state(args.random_state)
    if reproduce_random_state:
        if shuffle or args.n_train_splits > 1:
            log.warning(
                f"Passed {reproduce_random_state=} but {shuffle=} and {args.n_train_splits=} > 1."
                "Note that shuffling and random partitioning across nodes is currently not reproduced, "
                "i.e. the values will be the same but shuffled differently."
            )
        reproduce_random_state_before_useless_features(
            random_state=random_state,
            n_samples=args.n_samples,
            n_classes=args.n_classes,
            n_clusters_per_class=args.n_clusters_per_class,
            n_informative=dataset_config["make_classification_kwargs"]["n_informative"],
            n_redundant=dataset_config["make_classification_kwargs"]["n_redundant"],
        )
    log.debug(f"Current pos of random state: {random_state.get_state()[2]}")

    # Check for useless features and add where missing
    log.info("Start checking each slice for useless features and add where missing.")
    for group_name in [
        f"local_train_sets/{name}" for name in file["local_train_sets"]
    ] + ["test_set"]:
        log.debug(
            f"Adding useless features to {group_name}. Current random state pos: {random_state.get_state()[2]}"
        )
        group = file[group_name]

        if len(group["x"].shape) != 2 or group["x"].shape[1] not in [
            n_useful,
            n_features,
        ]:
            raise ValueError(
                f"Unexpected feature shape {group['x'].shape} in group {group_name}."
                f"Expected either only useful features (shape (_, {n_useful})) "
                f"or useful and useless features (shape (_, {n_features}))."
            )

        samples, features = group["x"].shape
        if features == n_useful:  # only useful features, add as before
            log.debug("Missing useless features, adding useless features now.")
            add_useless_features_to_hdf5(
                file, group_name, random_state, n_useless, shuffle
            )
        elif (
            features == n_features
        ):  # already has useful features, only simulate random state
            log.debug("Already contains all features, incrementing random state.")
            reproduce_random_state_add_useless_features(
                samples, n_useful, n_useless, random_state, shuffle
            )
    log.info("Done adding useless features.")


if __name__ == "__main__":
    set_logger_config(level=logging.DEBUG)
    args = parse_arguments()
    log.info(
        f"Train-split: {args.train_split} ({args.train_split * args.n_samples} train samples, "
        f"{(1 - args.train_split) * args.n_samples} train samples)"
    )
    if args.low_mem_data_generation:
        if args.continue_data_generation:
            log.info("Continuing memory efficient dataset generation.")
            continue_memory_efficient_dataset_generation(args)
        else:
            log.info("Generating dataset with memory efficient approach.")
            generate_and_save_dataset_memory_efficient(args)
    else:
        log.info("Generating dataset with standard make_classification approach.")
        generate_and_save_dataset(args)
