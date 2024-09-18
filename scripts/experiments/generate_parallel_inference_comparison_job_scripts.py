import os
import pathlib
import subprocess


def generate_parallel_inference_comparison_job_scripts(
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
    Generate the job scripts for the inference flavor comparison experiments.

    These experiments basically correspond to a weak scaling experiment series with shared global model.

    NOTE: We estimated 1500 and 450 trees to be trainable in serial in 3 days for 1M samples with 10k features and 10M
    samples with 1k features, respectively, and chose the closest number evenly divisible by 64 as a baseline.
    With number of samples n, number of features m, and number of trees t:

    Strong scaling:
    n6m4 baseline (n, m, t) = (10^6, 10^4, 1600) and n7m3 baseline: (n, m, t) = (10^7, 10^3, 448)

    Weak scaling:
    n6m4 baseline (n, m, t) = (10^6, 10^4, 800) and n7m3 baseline: (n, m, t) = (10^7, 10^3, 224)

    NOTE: All strong-scaling experiments used high-memory nodes, i.e., #SBATCH --mem=486400mb, except for the 64-node
    experiment which used the normal nodes. This is due to the fact that HoreKa has only 32 high-memory nodes. However,
    as the problem size per node decreases with increasing number of nodes in strong scaling, this was not a problem here
    but only for weak scaling. That is why the base problem size of weak scaling is only half the base problem size of
    strong scaling.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples to use.
    log_n_features : int
        The common logarithm of the number of features to use.
    n_classes : int
        The number of classes in the synthetic classification dataset.
    n_trees : int
        The number of trees to use in the baseline (will be scaled up with the number of nodes).
    data_seed : int
        The random state used for synthetic dataset generation and splitting.
    model_seed : int
        The (base) random state used for initializing the (distributed) model.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    for n_nodes in [
        2,
        4,
        8,
        16,
        32,
        64,
    ]:  # Weak scaling type experiment (with shared global model)
        n_trees_global = n_trees * n_nodes  # Number of trees is scaled with number of nodes.
        time = 3600  # All experiments should take approx. the same time (in min).
        mem = 243200  # Use standard nodes.

        print(
            f"Current config uses {n_nodes} nodes and {n_trees_global} trees. Wall-clock time is {time / 60}h."
        )

        job_name = (
            f"n{log_n_samples}_m{log_n_features}_nodes_{n_nodes}_modelseed_{model_seed}"
        )
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
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi/4.1
source "${{BASE_DIR}}"/special-couscous-venv-openmpi4/bin/activate  # Activate venv.

SCRIPT="special-couscous/scripts/examples/rf_parallel_synthetic.py"

RESDIR="${{BASE_DIR}}"/results/shared_global_model/n{log_n_samples}_m{log_n_features}/nodes_{n_nodes}/${{SLURM_JOB_ID}}_{data_seed}_{model_seed}/
mkdir -p "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{BASE_DIR}}/${{SCRIPT}} \\
    --n_samples {10**log_n_samples} \\
    --n_features {10**log_n_features} \\
    --n_classes {n_classes} \\
    --n_trees {n_trees_global} \\
    --random_state {data_seed} \\
    --random_state_model {model_seed} \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --detailed_evaluation \\
    --save_model \\
    --shared_global_model
                                """

        with open(output_path / job_script_name, "wt") as f:
            f.write(script_content)
        if submit:
            subprocess.run(f"sbatch {output_path}/{job_script_name}", shell=True)


if __name__ == "__main__":
    data_sets = [(6, 4, 800), (7, 3, 224)]
    data_seed = 0
    model_seeds = [1, 2, 3]
    n_classes = 10
    output_path = pathlib.Path("./inference_flavor/")
    os.makedirs(output_path, exist_ok=True)
    for random_state_model in model_seeds:
        for data_set in data_sets:
            log_n_samples = data_set[0]
            log_n_features = data_set[1]
            n_trees = data_set[2]
            # Generate job scripts and possibly submit them to the cluster.
            generate_parallel_inference_comparison_job_scripts(
                log_n_samples=log_n_samples,
                log_n_features=log_n_features,
                n_trees=n_trees,
                n_classes=n_classes,
                data_seed=data_seed,
                model_seed=random_state_model,
                output_path=output_path,
                submit=False,
            )
