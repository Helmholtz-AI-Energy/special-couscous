import time
import os.path
import numpy as np
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# FUNCTION DEFINITIONS
def load_data(
    data_path: str,
    header_lines: int,
    train_split: float = 0.75,
    sep: str = ",",
    target_first: bool = True,
    random_state: int = 42
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load data from CSV file.

    Parameters
    ----------
    data_path : str
                path to .csv file
    header_lines : int
                   number of header lines
    train_split : float
                  train-test split fraction
    sep : char
          character used in file to separate entries
    target_first : bool
                   True if targets are in the first column, False if in the last.
    random_state : int
                   seed for sklearn's train-test split

    Returns
    -------
    train_samples : numpy.ndarray
                    train samples
    train_targets : numpy.ndarray
                    train targets
    test_samples : numpy.ndarray
                   test samples
    test_targets : numpy.ndarray
                   test targets
    """
    # Load data into numpy array.
    data = np.loadtxt(data_path, dtype=float, delimiter=sep, skiprows=header_lines)
    # Divide data into samples and targets.
    if target_first:  # Targets in the first column.
        samples = data[:, 1:]
        targets = data[:, 0]
    else:  # Targets in the last column.
        samples = data[:, :-1]
        targets = data[:, -1]
    # Perform train-test split.
    samples_train, samples_test, targets_train, targets_test = train_test_split(
        samples, targets, test_size=1 - train_split, random_state=random_state
    )
    return samples_train, targets_train, samples_test, targets_test


if __name__ == "__main__":
    # SETTINGS

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        prog="Distributed Random Forests",
        description="Set up and distributed random forest classification.",
        # epilog="Add help text here.",
        )
    parser.add_argument("-rss", "--random_state_split", type=int)
    parser.add_argument("-t", "--n_trees", type=int)
    parser.add_argument("-rsf", "--random_state_forest", type=int)
    args = parser.parse_args()

    config = {
        "header_lines": 0,
        "sep": ",",
        "data_path": "/pfs/work7/workspace/scratch/ku4408-RandomForest/data/SUSY.csv",
        "target_first": True,
        "train_split": 0.75,
        "job_id": int(os.getenv("SLURM_JOB_ID")),
        "random_state_split": args.random_state_split,
        "random_state_forest": args.random_state_forest,
        "n_trees": args.n_trees,
    }
    print(
        f"Loading data...\n"
        f"Train-test split: Train fraction is {config['train_split']}, "
        f"random state is {config['random_state_split']}."
    )
    # Load data.
    start_load = time.perf_counter()
    (
        train_samples,
        train_targets,
        test_samples,
        test_targets,
    ) = load_data(
        config["data_path"],
        config["header_lines"],
        config["train_split"],
        config["sep"],
        config["random_state_split"],
    )
    elapsed_load = time.perf_counter() - start_load
    print(f"Done\nTrain samples and targets have shapes {train_samples.shape} and {train_targets.shape}.\n"
          f"First three elements are: {train_samples[:3]} and {train_targets[:3]}\n"
          f"Test samples and targets have shapes {test_samples.shape} and {test_targets.shape}.\n"
          f"First three elements are: {test_samples[:3]} and {test_targets[:3]}\n"
          f"Time for data loading is {elapsed_load} s."
          f"Set up classifier with {config['n_trees']} trees and random state {config['random_state_forest']}."
          )

    # Set up, train, and test model.
    clf = RandomForestClassifier(
        n_estimators=config["n_trees"],
        random_state=config["random_state_forest"]
    )
    start_train = time.perf_counter()
    print("Train.")
    clf.fit(train_samples, train_targets)
    pickle.dump(clf, open(f"./RF{config['n_trees']}-1_{config['job_id']}.pt", "wb"))
    acc = clf.score(test_samples, test_targets)
    elapsed_train = time.perf_counter() - start_train
    print(f"Time for training is {elapsed_train} s.\nAccuracy is {acc}.")
