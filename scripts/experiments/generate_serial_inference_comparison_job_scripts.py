import os
import pathlib
import subprocess


def generate_serial_inference_comparison_job_scripts(
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
    Generate the job scripts for the serial inference comparison baseline.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples in the synthetic dataset.
    log_n_features : int
        The common logarithm of the number of features in the synthetic dataset.
    n_classes : int
        The number of classes in the synthetic dataset.
    n_trees : int
        The number of trees to use.
    data_seed : int
        The random state to use for synthetic dataset generation and splitting.
    model_seed : int
        The random state to use for initializing the model.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    wall_time = 3600  # All experiments should take approx. the same time (in min).
    job_name = f"n{log_n_samples}_m{log_n_features}_nodes_1_modelseed_{model_seed}"
    job_script_name = f"{job_name}.sh"
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}  # Job name
#SBATCH --partition=cpuonly    # Queue for resource allocation
#SBATCH --nodes=1              # Number of nodes to use
#SBATCH --ntasks-per-node=1    # Number of tasks to use
#SBATCH --time={wall_time}     # Wall-clock time limit
#SBATCH --cpus-per-task=76     # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL        # Notify user by email when certain event types occur

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-SpecialCouscous}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi/4.1
source "${{BASE_DIR}}"/special-couscous-venv-openmpi4/bin/activate  # Activate venv.

SCRIPT="special-couscous/scripts/examples/rf_serial_synthetic.py"

RESDIR="${{BASE_DIR}}"/results/shared_global_model/n{log_n_samples}_m{log_n_features}/nodes_{1}/${{SLURM_JOB_ID}}_{data_seed}_{model_seed}/
mkdir -p "${{RESDIR}}"
cd "${{RESDIR}}" || exit

python -u ${{BASE_DIR}}/${{SCRIPT}} \\
    --n_samples {10**log_n_samples} \\
    --n_features {10**log_n_features} \\
    --n_trees {n_trees} \\
    --n_classes {n_classes} \\
    --random_state {data_seed} \\
    --random_state_model {model_seed} \\
    --detailed_evaluation \\
    --save_model \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --log_path ${{RESDIR}}
                                    """
    # Write script content to file.
    with open(output_path / job_script_name, "wt") as f:
        f.write(script_content)
    # Possibly submit script to cluster.
    if submit:
        subprocess.run(f"sbatch {output_path}/{job_script_name}", shell=True)


if __name__ == "__main__":
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
                generate_serial_inference_comparison_job_scripts(
                    log_n_samples=log_n_samples,
                    log_n_features=log_n_features,
                    n_trees=n_trees,
                    n_classes=n_classes,
                    data_seed=data_seed,
                    model_seed=random_state_model,
                    output_path=output_path,
                    submit=False,
                )
