import argparse
import pathlib
from typing import cast

from specialcouscous.scaling_dataset import read_scaling_dataset_from_hdf5
from specialcouscous.synthetic_classification_data import SyntheticDataset


def load_and_check_dataset(dataset: str, n_ranks: int = 64) -> None:
    """
    Load and check a scaling dataset by loading the HDF5 file rank by rank and printing the dataset sizes.

    Parameters
    ----------
    dataset : str
        Name of the baseline single-node dataset (n6m4 or n7m3).
    n_ranks : int
        The maximum number of ranks, the check will load slices range(n_ranks) from the HDF5. Default is 64.
    """
    dataset_dirnames = {
        "n6m4": "n_samples_48250000__n_features_10000__n_classes_10",
        "n7m3": "n_samples_482500000__n_features_1000__n_classes_10",
    }

    dirname = dataset_dirnames[dataset]
    hdf5_path = f"/hkfs/work/workspace/scratch/ku4408-SpecialCouscous/datasets/{dirname}/64_ranks__seed_0.h5"
    print(f"{hdf5_path=}\n")

    for rank in range(n_ranks):
        local_train_set, global_test_set, root_attrs = read_scaling_dataset_from_hdf5(
            pathlib.Path(hdf5_path), rank=rank, with_global_test=(rank == 0)
        )
        local_train_set = cast(SyntheticDataset, local_train_set)

        if rank == 0:
            print(f"Root attrs:\n{root_attrs}\n")
            global_test_set = cast(SyntheticDataset, global_test_set)
            print(
                f"Global test set: {global_test_set.n_samples} samples "
                f"with {global_test_set.x.shape[1]} features "
                f"and {global_test_set.n_classes} classes.\n"
            )

        print(f"Rank {rank:2d}: Local train set: {local_train_set.n_samples} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["n6m4", "n7m3"],
        help="Select the scaling dataset to check based on the name of it's single-node baseline",
    )
    args = parser.parse_args()
    load_and_check_dataset(args.dataset)
