import argparse
import os
import pickle
import time
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from synthetic_classification_data import make_classification_dataset

if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        prog="Random Forest",
        description="Generate synthetic classification data and classify with random forest.",
    )
    parser.add_argument("-s", "--n_samples", type=int)
    parser.add_argument("-f", "--n_features", type=int)
    parser.add_argument("-fi", "--frac_informative", type=float, default=0.1)
    parser.add_argument("-fr", "--frac_redundant", type=float, default=0.1)
    parser.add_argument("-c", "--n_classes", type=int, default=10)
    parser.add_argument("-cc", "--n_clusters_per_class", type=int, default=1)
    parser.add_argument("-rsg", "--random_state_generation", type=int, default=17)
    parser.add_argument("-ts", "--train_split", type=float, default=0.75)
    parser.add_argument("-rss", "--random_state_split", type=int, default=9)
    parser.add_argument("-t", "--n_trees", type=int)
    parser.add_argument("-rsf", "--random_state_forest", type=int, default=5)
    args = parser.parse_args()

    print(
        "**************************************************************\n"
        "* Single-Node Random Forest Classification of Synthetic Data *\n"
        "**************************************************************\n"
        f"Hyperparameters used are:\n{args}"
    )

    if os.getenv("SLURM_JOB_ID"):
        job_id = int(os.getenv("SLURM_JOB_ID"))
    else:
        job_id = 0  # Get SLURM job ID.

    pickle.dump(args, open(f"./config_{job_id}.pickle", "wb"))

    print(f"Generating data...")
    # Generate data.
    (
        train_samples,
        test_samples,
        train_targets,
        test_targets,
    ) = make_classification_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        frac_informative=args.frac_informative,
        frac_redundant=args.frac_redundant,
        n_classes=args.n_classes,
        n_clusters_per_class=args.n_clusters_per_class,
        random_state_generation=args.random_state_generation,
        train_split=args.train_split,
        random_state_split=args.random_state_split,
    )
    print(
        f"Done\nTrain samples and targets have shapes {train_samples.shape} and {train_targets.shape}.\n"
        f"First three elements are: {train_samples[:3]} and {train_targets[:3]}\n"
        f"Test samples and targets have shapes {test_samples.shape} and {test_targets.shape}.\n"
        f"First three elements are: {test_samples[:3]} and {test_targets[:3]}\n"
        f"Set up classifier."
    )

    # Set up, train, and test model.
    classifier = RandomForestClassifier(
        n_estimators=args.n_trees, random_state=args.random_state_forest
    )
    print("Train.")
    start_train = time.perf_counter()
    classifier.fit(train_samples, train_targets)
    elapsed_train = time.perf_counter() - start_train
    accuracy = classifier.score(test_samples, test_targets)
    print(f"Time for training is {elapsed_train} s.\nAccuracy is {accuracy}.")
