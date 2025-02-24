import os
import pathlib
import subprocess

from specialcouscous.utils.slurm import find_checkpoint_dir_and_uuid


def generate_parallel_evaluation_from_breaking_iid_ckpt_job_scripts(
    log_n_samples: int,
    log_n_features: int,
    n_classes: int,
    mu_global: float | str,
    mu_local: float | str,
    n_trees: int,
    data_seed: int,
    model_seed: int,
    output_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    checkpoint_uid: str,
    submit: bool = False,
) -> None:
    """
    Generate the job scripts for parallel evaluation of breaking-IID experiments from pickled model checkpoints.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples to use.
    log_n_features : int
        The common logarithm of the number of features to use.
    n_classes : int
        The number of classes in the synthetic classification dataset.
    mu_global : float | str
        The global imbalance factor.
    mu_local : float | str
        The local imbalance factor.
    n_trees : int
        The number of trees to use in the baseline (will be scaled up with the number of nodes).
    data_seed : int
        The random state used for synthetic dataset generation and splitting.
    model_seed : int
        The (base) random state used for initializing the (distributed) model.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    checkpoint_path : pathlib.Path
        The path to load the pickled model checkpoints from.
    checkpoint_uid : str
        The considered run's unique identifier. Used to identify the correct checkpoints to load.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    mem = 243200  # Use standard nodes.
    n_nodes = 16  # Breaking-IID experiments all use 16 nodes.
    time = 120  # Runtime estimated from trial runs

    print(
        f"Current config uses {n_nodes} nodes and {n_nodes * n_trees} trees. Wall-clock time is {time / 60}h."
    )

    job_name = f"n{log_n_samples}_m{log_n_features}_nodes_{n_nodes}_{data_seed}_{model_seed}_{str(mu_global).replace('.', '')}_{str(mu_local).replace('.', '')}"
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

SCRIPT="special-couscous/scripts/examples/evaluate_from_checkpoint_breaking_iid.py"

RESDIR="${{BASE_DIR}}"/results/breaking_iid/n{log_n_samples}_m{log_n_features}/nodes_{n_nodes}/${{SLURM_JOB_ID}}_{data_seed}_{model_seed}_{str(mu_global).replace(".", "")}_{str(mu_local).replace(".", "")}/
mkdir -p "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun --mpi=pmix python -u ${{BASE_DIR}}/${{SCRIPT}} \\
    --n_samples {10**log_n_samples} \\
    --n_features {10**log_n_features} \\
    --n_classes {n_classes} \\
    --shared_test_set \\
    --globally_imbalanced \\
    --mu_data {mu_global} \\
    --locally_imbalanced \\
    --mu_partition {mu_local} \\
    --random_state {data_seed} \\
    --checkpoint_path {checkpoint_path} \\
    --checkpoint_uid {checkpoint_uid} \\
    --n_trees {n_nodes * n_trees} \\
    --random_state_model {model_seed} \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}}
                                """
    output_path = output_path / f"nodes_{n_nodes}"
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / job_script_name, "wt") as f:
        f.write(script_content)
    if submit:
        subprocess.run(f"sbatch {output_path}/{job_script_name}", shell=True)


if __name__ == "__main__":
    data_sets = [
        (6, 4, 800),
        (7, 3, 224),
    ]  # Baseline problem as (`log_n_samples`, `log_n_features`, `n_trees`)
    data_seeds = [0]  # , 1, 2]  # Data seed to use
    model_seeds = [0, 1, 2]  # Model seeds to use
    n_classes = 10  # Number of classes to use
    mu_global = [0.5, 2.0, "inf"]  # Global imbalance factors considered
    mu_local = [0.5, 2.0, "inf"]  # Local imbalance factors considered
    output_path = pathlib.Path(
        "./evaluate_breaking_iid_from_ckpt/"
    )  # Output path to save generated job scripts
    os.makedirs(output_path, exist_ok=True)
    base_path = pathlib.Path(
        "/hkfs/work/workspace/scratch/ku4408-SpecialCouscous/results/"
    )  # Base path to find correct checkpoint path
    # Loop over all considered configurations.
    for random_state_data in data_seeds:
        for random_state_model in model_seeds:
            for data_set in data_sets:
                for m_global in mu_global:
                    for m_local in mu_local:
                        log_n_samples = data_set[0]
                        log_n_features = data_set[1]
                        n_trees = data_set[2]
                        # Generate job scripts and possibly submit them to the cluster.
                        assert (
                            isinstance(m_global, str) or isinstance(m_global, float)
                        ) and (isinstance(m_local, str) or isinstance(m_local, float))
                        checkpoint_path, checkpoint_uid = find_checkpoint_dir_and_uuid(
                            base_path=base_path,
                            log_n_samples=log_n_samples,
                            log_n_features=log_n_features,
                            mu_global=m_global,
                            mu_local=m_local,
                            data_seed=random_state_data,
                            model_seed=random_state_model,
                        )
                        generate_parallel_evaluation_from_breaking_iid_ckpt_job_scripts(
                            log_n_samples=log_n_samples,
                            log_n_features=log_n_features,
                            n_classes=n_classes,
                            mu_global=m_global,
                            mu_local=m_local,
                            n_trees=n_trees,
                            data_seed=random_state_data,
                            model_seed=random_state_model,
                            output_path=output_path,
                            checkpoint_path=checkpoint_path,
                            checkpoint_uid=checkpoint_uid,
                            submit=False,
                        )
