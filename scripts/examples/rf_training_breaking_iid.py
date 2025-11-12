#!/usr/bin/env python
"""Distributed random forest classification of a possibly imbalanced synthetic dataset in `specialcouscous``."""

import logging
import pathlib

from mpi4py import MPI

from specialcouscous.train.train_parallel import train_parallel_on_synthetic_data
from specialcouscous.utils import parse_arguments, set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.

if __name__ == "__main__":
    # Parse command-line arguments.
    args = parse_arguments()

    # Set up separate logger for ``specialcouscous``.
    set_logger_config(
        level=args.logging_level,  # Logging level
        log_file=f"{args.log_path}/{pathlib.Path(__file__).stem}.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    if MPI.COMM_WORLD.rank == 0:
        log.info(
            "**************************************************************\n"
            "* Distributed Random Forest Classification of Synthetic Data *\n"
            "**************************************************************\n"
            f"Hyperparameters used are:\n{args}"
        )
    # Train distributed random forest on synthetic classification data.
    train_parallel_on_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        globally_balanced=not args.globally_imbalanced,
        locally_balanced=not args.locally_imbalanced,
        shared_test_set=args.shared_test_set,
        random_state=args.random_state,
        random_state_model=args.random_state_model,
        mu_partition=args.mu_partition,
        mu_data=args.mu_data,
        peak=args.peak,
        enforce_constant_local_size=args.enforce_constant_local_size,
        make_classification_kwargs={
            "n_clusters_per_class": args.n_clusters_per_class,
            "n_informative": int(args.frac_informative * args.n_features),
            "n_redundant": int(args.frac_redundant * args.n_features),
            "flip_y": args.flip_y,
        },
        comm=MPI.COMM_WORLD,
        train_split=args.train_split,
        stratified_train_test=args.stratified_train_test,
        n_trees=args.n_trees,
        shared_global_model=args.shared_global_model,
        detailed_evaluation=args.detailed_evaluation,
        output_dir=args.output_dir,
        output_label=args.output_label,
        experiment_id=args.experiment_id,
        save_model=args.save_model,
    )
