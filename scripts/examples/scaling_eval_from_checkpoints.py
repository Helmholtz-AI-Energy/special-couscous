import argparse
import logging
import pathlib
from typing import Any

import numpy as np
import pandas
from mpi4py import MPI
from sklearn.utils.validation import check_random_state

from specialcouscous.rf_parallel import DistributedRandomForest
from specialcouscous.synthetic_classification_data import (
    generate_and_distribute_synthetic_dataset,
)
from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


def evaluate_scaling_data_and_model_from_chunking_checkpoints(
    n_ranks: int,
    checkpoint_path: str | pathlib.Path,
    checkpoint_uid: str,
    n_trees: int,
    output_dir: pathlib.Path | str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    make_classification_kwargs: dict[str, Any] | None = None,
    random_state: int | np.random.RandomState = 0,
    train_split: float = 0.75,
    stratified_train_test: bool = False,
) -> None:
    """
    Evaluate distributed random forest from checkpoints while simulating growing data and model size.

    Compute the accuracy for a p-rank "global" model using the first p local models. By using local models trained with
    chunking, this simulates simultaneously growing both data and model size.

    The evaluation is performed serially (one node loads all checkpoints and performs inference of the local model
    sequentially). It is not necessary (or useful) to call this function with more than one MPI rank. Use the n_ranks
    parameter to specify the number of ranks to simulate (i.e. the maximum p to evaluate).

    Parameters
    ----------
    n_ranks : int
        The number of ranks to simulate (i.e. the maximum number of available checkpoints).
    checkpoint_path : pathlib.Path | str
        The directory containing the pickled local model checkpoints to load.
    checkpoint_uid : str
        The considered run's unique identifier. Used to identify the correct checkpoints to load.
    n_trees : int
        The number of trees in the global forest.
    output_dir : pathlib.Path | str
        Output base directory.
    n_samples : int
        The number of samples in the dataset.
    n_features : int
        The number of features in the dataset.
    n_classes : int
        The number of classes in the dataset.
    make_classification_kwargs : dict[str, Any], optional
        Additional keyword arguments to ``sklearn.datasets.make_classification``.
    random_state : int | np.random.RandomState
        The random seed, used for dataset generation, partition, and distribution. Can be  an integer or a numpy random
        state as it must be the same on all ranks to ensure that each rank generates the very same global dataset.
    train_split : float
        Relative size of the train set.
    stratified_train_test : bool
        Whether to stratify the train-test split with the class labels. Default is False.
    """
    configuration = locals()
    random_state = check_random_state(random_state)
    output_dir = pathlib.Path(output_dir)
    fake_comm = MPI.COMM_WORLD
    # -------------- Generate data as during training, keep only test data --------------
    log.info("Generating synthetic data.")
    (
        _,
        _,
        test_set,
    ) = generate_and_distribute_synthetic_dataset(
        globally_balanced=True,
        locally_balanced=True,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        rank=0,
        n_ranks=n_ranks,
        random_state=random_state,
        test_size=1 - train_split,
        make_classification_kwargs=make_classification_kwargs,
        shared_test_set=True,
        stratified_train_test=stratified_train_test,
    )
    log.debug("Data generation done.")

    # -------------- Load model checkpoints one by one and collect histograms --------------
    local_histograms_by_rank = []
    for rank in range(n_ranks):
        # load trained local model for rank <rank> from checkpoint
        log.info(f"Loading classifier {rank} / {n_ranks} from checkpoint.")
        local_subforest = DistributedRandomForest(
            n_trees_global=n_trees, comm=fake_comm
        )
        local_subforest.load_checkpoints(checkpoint_path, checkpoint_uid, rank=rank)
        if len(local_subforest.clf.estimators_) != (n_trees / n_ranks):
            log.warning(
                f"Expected {n_trees // n_ranks} trees per local forest, "
                f"but got {len(local_subforest.clf.estimators_)}."
            )
        log.debug("Classifier loaded successfully.")

        # compute histogram predictions for test set
        log.info(f"Computing local histograms for {rank=} on the global test set.")
        sample_wise_predictions = local_subforest._predict_locally(
            test_set.x
        ).transpose()
        log.debug(
            f"sample_wise_predictions have shape {sample_wise_predictions.shape}."
        )
        local_histograms = np.array(
            [
                np.bincount(sample_pred, minlength=n_classes)
                for sample_pred in sample_wise_predictions
            ]
        )
        log.debug(f"local_histograms have shape {local_histograms.shape}.")
        local_histograms_by_rank.append(local_histograms)

        # save histogram to disk
        histogram_path = output_dir / "local_histograms" / f"histograms_rank_{rank}.csv"
        histogram_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving local histograms to {histogram_path}.")
        np.savetxt(histogram_path, local_histograms, delimiter=",")

    # -------- Aggregate "global" histograms using different local model counts and compute accuracy --------
    global_histograms = []
    current_global_histograms = np.zeros_like(local_histograms_by_rank[0])
    results = []
    for rank, local_histograms in enumerate(local_histograms_by_rank):
        log.debug(f"Computing global histogram using ranks 0..{rank}.")
        current_global_histograms += local_histograms
        log.debug(
            f"current_global_histograms have shape {current_global_histograms.shape}."
        )
        global_histograms.append(current_global_histograms)
        predictions = np.array(
            [np.argmax(sample_hist) for sample_hist in current_global_histograms]
        )
        log.debug(f"predictions have shape {predictions.shape}.")
        accuracy = (test_set.y == predictions).mean()
        comm_size = rank + 1
        results.append(
            {
                **configuration,
                "comm_size": comm_size,
                "accuracy_global_test": accuracy,
                "n_trees": n_trees // n_ranks * comm_size,
            }
        )
        log.info(
            f"Global test accuracy using the first {rank:2d} models is: {accuracy:7.2%}."
        )

    # -------------- Save results as csv --------------
    results_path = output_dir / "scaling_data_and_mode_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving results to {results_path}")
    results_df = pandas.DataFrame(results)
    results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    set_logger_config()

    dataset_configs = {  # n_samples, n_features, n_trees
        "n5m3": (int(1e5), int(1e3), 800),
        "n6m4": (int(1e6), int(1e4), 1600),
        "n7m3": (int(1e7), int(1e3), 448),
        "n6m4_64th": (15625, int(1e4), 1600),
        "n7m3_64th": (156250, int(1e3), 448),
    }
    available_datasets = list(dataset_configs.keys())

    chunking_checkpoint_base_path = pathlib.Path(
        "/hkfs/work/workspace/scratch/ku4408-specialcouscous-eps/chunking/"
    )
    chunking_checkpoint_path_n6m4 = chunking_checkpoint_base_path / "n6_m4/nodes_64"
    chunking_checkpoint_path_n7m3 = chunking_checkpoint_base_path / "n7_m3/nodes_64"

    no_chunking_checkpoint_base_path = pathlib.Path(
        "/hkfs/work/workspace/scratch/bk6983-special_couscous__2025_results/scaling_the_model/"  # pragma: allowlist secret
    )

    # (n_samples, n_features, n_trees, seed_data, seed_model, n_ranks) -> (checkpoint_path, checkpoint_uid)
    checkpoints = {
        # n5m3 test run
        (100000, 1000, 800, 0, 1, 8): (
            pathlib.Path(__file__).parents[2]
            / "results/2025/2025-3/2025-03-14/chunking_n5m3",
            "12a1bad6",
        ),
        # n6m4
        (1000000, 10000, 1600, 0, 1, 64): (
            chunking_checkpoint_path_n6m4 / "2714948_0_1",
            "25c1dc56",
        ),
        (1000000, 10000, 1600, 0, 2, 64): (
            chunking_checkpoint_path_n6m4 / "2714960_0_2",
            "e787869f",
        ),
        (1000000, 10000, 1600, 0, 3, 64): (
            chunking_checkpoint_path_n6m4 / "2714972_0_3",
            "4c2bda80",
        ),
        (1000000, 10000, 1600, 1, 1, 64): (
            chunking_checkpoint_path_n6m4 / "2714984_1_1",
            "6c95e50b",
        ),
        (1000000, 10000, 1600, 1, 2, 64): (
            chunking_checkpoint_path_n6m4 / "2714996_1_2",
            "4dda6383",
        ),
        (1000000, 10000, 1600, 1, 3, 64): (
            chunking_checkpoint_path_n6m4 / "2715008_1_3",
            "efee63da",
        ),
        (1000000, 10000, 1600, 2, 1, 64): (
            chunking_checkpoint_path_n6m4 / "2715020_2_1",
            "b71f95a0",
        ),
        (1000000, 10000, 1600, 2, 2, 64): (
            chunking_checkpoint_path_n6m4 / "2715032_2_2",
            "69ab4e98",
        ),
        (1000000, 10000, 1600, 2, 3, 64): (
            chunking_checkpoint_path_n6m4 / "2715044_2_3",
            "4d9d6b4b",
        ),
        # n67m3
        (10000000, 1000, 448, 0, 1, 64): (
            chunking_checkpoint_path_n7m3 / "2714954_0_1",
            "d28793fc",
        ),
        (10000000, 1000, 448, 0, 2, 64): (
            chunking_checkpoint_path_n7m3 / "2714966_0_2",
            "c672c865",
        ),
        (10000000, 1000, 448, 0, 3, 64): (
            chunking_checkpoint_path_n7m3 / "2714978_0_3",
            "36c83d13",
        ),
        (10000000, 1000, 448, 1, 1, 64): (
            chunking_checkpoint_path_n7m3 / "2714990_1_1",
            "57b532d0",
        ),
        (10000000, 1000, 448, 1, 2, 64): (
            chunking_checkpoint_path_n7m3 / "2715002_1_2",
            "9c975e89",
        ),
        (10000000, 1000, 448, 1, 3, 64): (
            chunking_checkpoint_path_n7m3 / "2715014_1_3",
            "021295b2",
        ),
        (10000000, 1000, 448, 2, 1, 64): (
            chunking_checkpoint_path_n7m3 / "2715026_2_1",
            "4f5bee85",
        ),
        (10000000, 1000, 448, 2, 2, 64): (
            chunking_checkpoint_path_n7m3 / "2715038_2_2",
            "f694b9b1",
        ),
        (10000000, 1000, 448, 2, 3, 64): (
            chunking_checkpoint_path_n7m3 / "2715050_2_3",
            "96481d50",
        ),
        # n6m4 64th
        (15625, 10000, 1600, 0, 1, 64): (
            no_chunking_checkpoint_base_path / "n6_m4/nodes_64" / "0_1_2992744",
            "398c1cb1",
        ),
        # n67m3 64th
        (156250, 1000, 448, 0, 1, 64): (
            no_chunking_checkpoint_base_path / "n7_m3/nodes_64" / "0_1_2992745",
            "9058eb20",
        ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ranks", type=int, default=64)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--random_state_model", type=int, default=1)
    parser.add_argument("--output_dir", type=pathlib.Path, default="./results")

    # specify either dataset or n_samples, n_features, and n_trees (overwritten otherwise)
    parser.add_argument("--dataset", type=str, choices=available_datasets)
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--n_features", type=int)
    parser.add_argument("--n_trees", type=int)
    parser.add_argument("--scale_n_samples", float=int, default=1.0)

    args = parser.parse_args()

    if args.dataset is not None:
        if args.dataset not in dataset_configs:
            raise ValueError(
                f"Invalid dataset {args.dataset}. Available options are: {available_datasets}."
            )
        args.n_samples, args.n_features, args.n_trees = dataset_configs[args.dataset]

    checkpoint_key = (
        args.n_samples,
        args.n_features,
        args.n_trees,
        args.random_state,
        args.random_state_model,
        args.n_ranks,
    )
    if checkpoint_key not in checkpoints:
        raise ValueError(f"No checkpoint available for {checkpoint_key}.")
    checkpoint_path, checkpoint_uid = checkpoints[checkpoint_key]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_path} not found for config {args}."
        )

    label = f"n{args.n_samples}_m{args.n_features}_t{args.n_trees}_s{args.random_state}_p{args.n_ranks}"
    output_dir = args.output_dir / "scaling_eval_from_chunking_checkpoints" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    make_classification_kwargs = {
        "n_clusters_per_class": 1,
        "n_informative": int(0.1 * args.n_features),
        "n_redundant": int(0.1 * args.n_features),
        "flip_y": 0.01,
    }

    evaluate_scaling_data_and_model_from_chunking_checkpoints(
        args.n_ranks,
        checkpoint_path,
        checkpoint_uid,
        args.n_trees,
        output_dir,
        int(args.n_samples * args.scale_n_samples),
        args.n_features,
        n_classes=10,
        make_classification_kwargs=make_classification_kwargs,
        random_state=args.random_state,
    )
