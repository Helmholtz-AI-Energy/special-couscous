import argparse
import os
import pathlib
from typing import Any

from specialcouscous.synthetic_classification_data import write_scaling_dataset_to_hdf5, generate_scaling_dataset, \
    read_scaling_dataset_from_hdf5
from specialcouscous.utils import parse_arguments


def dataset_path(
    root_path: os.PathLike, n_samples: int, n_features: int, n_classes: int, n_nodes: int, seed: int
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
    Wrapper for dataset_path which directly takes the args object from the CLI parser.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.

    Returns
    -------
    pathlib.Path
        The path to the HDF5 file for the corresponding dataset.
    """
    return dataset_path(
        root_path=args.data_root_path,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_nodes=args.n_nodes,
        seed=args.random_state,
    )


def dataset_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """
    Convert the CLI parameters to the configuration passed to generate_scaling_dataset.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.


    Returns
    -------
    dict[str, Any]
        The configuration parameters passed to generate_scaling_dataset.
    """
    return {
        'n_samples': args.n_samples,
        'n_features': args.n_features,
        'n_classes': args.n_classes,
        'n_ranks': args.n_train_splits,
        'random_state': args.random_state,
        'test_size': 1 - args.train_split,
        'make_classification_kwargs': {
            "n_clusters_per_class": args.n_clusters_per_class,
            "n_informative": int(args.frac_informative * args.n_features),
            "n_redundant": int(args.frac_redundant * args.n_features),
            "flip_y": args.flip_y,
        },
        'stratified_train_test': args.stratified_train_test,
    }


def load_and_verify_dataset(args, fail_on_unmatched_config=False):
    """
    Try to read the corresponding dataset from HDF5 and verify that the actual dataset config stored in the HDF5 matches
    the config inferred from the current CLI parameter.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.
    fail_on_unmatched_config : bool
        If true, an error is raised if the verification fails. Otherwise, only an error message is printed by the
        (potentially mismatched) dataset is returned anyway.

    Returns
    -------

    """
    # read dataset from HDF5
    path = dataset_path_from_args(args)
    local_train_sets, global_test_set, attrs = read_scaling_dataset_from_hdf5(path)

    # verify that the metadata stored within the HDF5 is identical to that specified by the parameters
    expected_dataset_config = dataset_config_from_args(args)
    actual_dataset_config = {key: value for key, value in attrs.items() if key in expected_dataset_config}

    if expected_dataset_config != actual_dataset_config:
        error_message = (f'Dataset config does not match current CLI arguments. '
                         f'From CLI {expected_dataset_config}, actual in HDF5 {actual_dataset_config}.')
        if fail_on_unmatched_config:
            raise ValueError(error_message)
        print(f'Warning: {error_message}')

    return local_train_sets, global_test_set, attrs


def generate_and_save_dataset(args: argparse.Namespace) -> None:
    """
    Generate a scaling dataset based on the given CLI parameters.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed CLI parameters.
    """
    # generate the dataset
    dataset_config = dataset_config_from_args(args)
    print(f'Creating dataset with the following parameters:\n{dataset_config}')
    global_train_set, local_train_sets, global_test_set = generate_scaling_dataset(**dataset_config)

    # write the dataset to HDF5
    path = dataset_path_from_args(args)
    write_scaling_dataset_to_hdf5(global_train_set, local_train_sets, global_test_set, dataset_config, path)
    print(f'Dataset successfully written to {path}.')
    print(f'To use this dataset, call \'scaling_dataset.load_and_verify_dataset(args)\' with the same CLI arguments.')


if __name__ == "__main__":
    args = parse_arguments()
    generate_and_save_dataset(args)
