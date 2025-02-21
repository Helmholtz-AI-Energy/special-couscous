import argparse
import os
import pathlib


def generate_job_script(
    n_nodes: int,
    log_n_samples_local: int,
    log_n_features: int,
    n_trees_local: int,
    data_seed: int,
    model_seed: int,
    output_path: pathlib.Path,
    n_classes: int = 10,
    train_split: float = 0.75,
    n_nodes_max: int = 64,
) -> None:
    """
    Generate the job scripts for the training experiments scaling model and data simultaneously.

    Based on the weak scaling experiments but using rf_parallel_synthetic_scale_data_and_model.py instead of
    rf_parallel_synthetic.py
    As for the weak scaling experiments, we can reconstruct the models for p ≤ 64 from the 64 node checkpoint. Since we
    don't expect a difference in how the training time scales compared to the other experiments, we only run the 64 node
    training runs, followed by multiple evaluation runs from the checkpoint for all p < 64.

    Parameters
    ----------
    n_nodes : int
        The number of nodes to run the training for.
    log_n_samples_local : int
        The common logarithm of the number of *local* samples to use (the train set will be scaled up to n_nodes).
    log_n_features : int
        The common logarithm of the number of features to use.
    n_trees_local : int
        The number of trees to use in the baseline (will be scaled up with the number of nodes).
    data_seed : int
        The random state used for synthetic dataset generation and splitting.
    model_seed : int
        The (base) random state used for initializing the (distributed) model.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    n_classes : int
        The number of classes in the synthetic classification dataset.
    train_split : float
        Fraction of local data in the train set. The remainder makes up the test set.
    n_nodes_max : int
        The maximum number of nodes (used to select the pre-generated dataset).
    """
    # Number of trees is scaled with number of nodes.
    n_trees_global = n_trees_local * n_nodes

    # Number of train samples is scaled with nodes, global test set remains unchanged
    n_samples_local = 10**log_n_samples_local
    n_samples_local_train = n_samples_local * train_split
    n_samples_global_test = n_samples_local * (1 - train_split)
    n_samples_global = int(n_samples_local_train * n_nodes_max + n_samples_global_test)

    actual_train_split = 1 - (n_samples_global_test / n_samples_global)

    print(
        f"Current config uses {n_nodes} nodes with {n_trees_global} trees and {n_samples_global} samples."
        f"Train split is {actual_train_split:.15f}."
    )

    job_name = f"n{log_n_samples_local}_m{log_n_features}_nodes_{n_nodes}_modelseed_{model_seed}"
    job_script_name = f"{job_name}.sh"

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}         # Job name
#SBATCH --partition=cpuonly           # Queue for resource allocation
#SBATCH --time=360                    # Wall-clock time limit (60h)
#SBATCH --mem=243200mb                # Main memory (full standard node)
#SBATCH --cpus-per-task=76            # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --nodes={n_nodes}             # Number of nodes
#SBATCH --ntasks-per-node=1           # One MPI rank per node
#SBATCH --account=hk-project-p0022229

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-SpecialCouscous}}  # root dir of the special-couscous workspace
SCRIPT_DIR="${{SCRIPT_DIR:-"${{BASE_DIR}}/special-couscous"}}"               # root of the special-couscous repository
DATA_DIR="${{BASE_DIR}}/datasets"                                            # dataset dir to write the generated datasets to
VENV=${{VENV:-${{BASE_DIR}}"/special-couscous-venv-openmpi4"}}               # path to the python venv to use
RESDIR="${{BASE_DIR}}"/results/scale_data_and_model/n{log_n_samples_local}_m{log_n_features}/nodes_{n_nodes}/${{SLURM_JOB_ID}}_{data_seed}_{model_seed}/

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

ml purge              # Unload all currently loaded modules.
ml load compiler/llvm  # Load required modules.
ml load mpi/openmpi/4.1
source "${{VENV}}"/bin/activate  # Activate venv.

SCRIPT="${{SCRIPT_DIR}}/scripts/examples/rf_parallel_synthetic_scale_data_and_model.py"

mkdir -p "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{SCRIPT}} \\
    --n_samples {n_samples_global} \\
    --n_features {10**log_n_features} \\
    --n_classes {n_classes} \\
    --n_trees {n_trees_global} \\
    --train_split {actual_train_split:.15f} \\
    --random_state {data_seed} \\
    --random_state_model {model_seed} \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --detailed_evaluation \\
    --save_model \\
    --data_root_path ${{DATA_DIR}} \\
    --n_train_splits {n_nodes_max}
"""

    script_path = output_path / job_script_name
    with open(script_path, "wt") as f:
        f.write(script_content)
        print(f"Script successfully written to {script_path.absolute()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Pass this to use the tiny test forest instead of the full scale forest",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=64,
        help="Number of nodes to run the training for.",
    )
    args = parser.parse_args()

    full_scale_data_sets = [(6, 4, 800), (7, 3, 224)]
    tiny_data_sets = [(6, 4, 10), (7, 3, 10)]
    data_sets = tiny_data_sets if args.tiny else full_scale_data_sets

    data_seed = 0
    model_seeds = [1] if args.tiny else [1, 2, 3]

    script_dir_name = "scale_data_and_model__train" + ("__tiny" if args.tiny else "")
    output_path = pathlib.Path(".") / script_dir_name
    os.makedirs(output_path, exist_ok=True)

    for model_seed in model_seeds:
        for log_n_samples, log_n_features, n_trees in data_sets:
            generate_job_script(
                args.n_nodes,
                log_n_samples,
                log_n_features,
                n_trees,
                data_seed,
                model_seed,
                output_path,
            )
