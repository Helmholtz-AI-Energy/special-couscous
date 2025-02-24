import os
import pathlib
import subprocess

limit = 60 * 24 * 3  # HoreKa wall-clock time limit in minutes
nodes = [2, 4, 8, 16, 32, 64]  # Number of nodes for scaling exps


def generate_chunking_job_scripts(
    log_n_samples: int,
    log_n_features: int,
    n_classes: int,
    n_trees: int,
    data_seed: int,
    model_seed: int,
    output_path: pathlib.Path,
    submit: bool = False,
) -> None:
    """
    Generate the job scripts for the strong scaling experiments with dataset chunking.

    NOTE: We estimated 1500 and 450 trees to be trainable in serial in 3 days for 1M samples with 10k features and 10M
    samples with 1k features, respectively, and chose the closest number evenly divisible by 64 as a baseline.
    With number of samples n, number of features m, and number of trees t:

    Strong scaling:
    n6m4 baseline (n, m, t) = (10^6, 10^4, 1600) and n7m3 baseline: (n, m, t) = (10^7, 10^3, 448)

    NOTE: All strong-scaling experiments used high-memory nodes, i.e., #SBATCH --mem=486400mb, except for the 64-node
    experiment which used the normal nodes. This is due to the fact that HoreKa has only 32 high-memory nodes. However,
    as the problem size per node decreases with increasing number of nodes in strong scaling, this was not a problem here
    but only for weak scaling.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples to use.
    log_n_features : int
        The common logarithm of the number of features to use.
    n_classes : int
        The number of classes to use.
    n_trees : int
        The number of trees to use. For weak scaling, this number serves as a baseline and is scaled up with the number
        of nodes.
    data_seed : int
        The random state used for synthetic dataset generation, splitting, and distribution.
    model_seed : int
        The (base) random state used for initializing the (distributed) model.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    for n_nodes in nodes:
        time = int(
            limit / n_nodes * 1.2
        )  # Run time should decrease with increasing number of nodes.
        mem = (
            486400 if n_nodes != 64 else 243200
        )  # Use high-memory nodes (except for 64-node experiment).

        print(
            f"Current config uses {n_nodes} nodes and {n_trees} trees. Wall-clock time is {time / 60}h."
        )

        job_name = f"n{log_n_samples}_m{log_n_features}_nodes_{n_nodes}_{data_seed}_{model_seed}"
        job_script_name = f"{job_name}.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}         # Job name
#SBATCH --partition=cpuonly           # Queue for resource allocation
#SBATCH --time={time}                 # Wall-clock time limit
#SBATCH --mem={mem}mb                 # Main memory
#SBATCH --cpus-per-task=76            # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --nodes={n_nodes}             # Number of nodes
#SBATCH --ntasks-per-node=1           # One MPI rank per node

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-SpecialCouscous}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

ml purge              # Unload all currently loaded modules.
ml load compiler/llvm  # Load required modules.
ml load mpi/openmpi/4.1
source "${{BASE_DIR}}"/special-couscous-venv-openmpi4/bin/activate  # Activate venv.

SCRIPT="special-couscous/scripts/examples/rf_training_breaking_iid.py"

RESDIR="${{BASE_DIR}}"/results/chunking/n{log_n_samples}_m{log_n_features}/nodes_{n_nodes}/${{SLURM_JOB_ID}}_{data_seed}_{model_seed}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{BASE_DIR}}/${{SCRIPT}} \\
    --n_samples {10**log_n_samples} \\
    --n_features {10**log_n_features} \\
    --n_classes {n_classes} \\
    --shared_test_set \\
    --n_trees {n_trees} \\
    --random_state {data_seed} \\
    --random_state_model {model_seed} \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --detailed_evaluation \\
    --save_model
                                """

        with open(output_path / job_script_name, "wt") as f:
            f.write(script_content)
        if submit:
            subprocess.run(f"sbatch {job_script_name}", shell=True)


if __name__ == "__main__":
    data_sets = [
        (6, 4, 1600),
        (7, 3, 448),
    ]  # Baseline problem as (`log_n_samples`, `log_n_features`, `n_trees`)
    data_seeds = [0, 1, 2]  # Data seed to use
    model_seeds = [3, 4, 5]  # Model seeds to use
    n_classes = 10  # Number of classes to use
    output_path = pathlib.Path(
        "./chunking/"
    )  # Output path to save generated job scripts
    os.makedirs(output_path, exist_ok=True)

    # Loop over all considered configurations.
    for random_state_data in data_seeds:
        for random_state_model in model_seeds:
            for data_set in data_sets:
                log_n_samples = data_set[0]
                log_n_features = data_set[1]
                n_trees = data_set[2]
                # Generate job scripts and possibly submit them to the cluster.
                generate_chunking_job_scripts(
                    log_n_samples=log_n_samples,
                    log_n_features=log_n_features,
                    n_classes=n_classes,
                    n_trees=n_trees,
                    data_seed=random_state_data,
                    model_seed=random_state_model,
                    output_path=output_path,
                    submit=False,
                )
