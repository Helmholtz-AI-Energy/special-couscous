#!/usr/bin/env python
"""Distributed random forest classification of a given dataset in `specialcouscous``."""

import logging
import pathlib

from mpi4py import MPI

from specialcouscous.train.train_parallel import train_parallel_on_dataset
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
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        log.info(
            "***********************************************************************\n"
            "* Distributed Random Forest Classification on A Given Dataset *\n"
            "***********************************************************************\n"
            f"Hyperparameters used are:\n{args}"
        )

    # Train distributed random forest on the given dataset.
    train_parallel_on_dataset(
        dataset=args.dataset_name,
        random_state=args.random_state,
        random_state_model=args.random_state_model,
        mpi_comm=comm,
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
