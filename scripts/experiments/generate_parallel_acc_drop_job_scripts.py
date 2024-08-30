import pathlib
import subprocess


def generate_parallel_acc_drop_job_scripts(
    log_n_samples: int,
    log_n_features: int,
    wall_time: int,
    n_classes: int,
    n_trees: int,
    n_tasks: int,
    random_state_data: int,
    random_state_model: int,
    output_path: pathlib.Path,
    submit: bool = False,
) -> None:
    """
    Generate the job scripts for the parallel dropping accuracy experiments.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples in the synthetic dataset.
    log_n_features : int
        The common logarithm of the number of features in the synthetic dataset.
    wall_time : int
        The wall-time in minutes.
    n_classes : int
        The number of classes in the synthetic dataset.
    n_trees : int
        The number of trees to use.
    n_tasks : int
        The number of tasks to use on a single node.
    random_state_data : int
        The random state.
    random_state_model : int
        Additional random state for the model.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    job_name = (
        f"n{log_n_samples}_m{log_n_features}_ntasks_{n_tasks}_dataseed_"
        f"{random_state_data}_modelseed_{random_state_model}"
    )
    job_script_name = f"{job_name}.sh"
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}         # Job name
#SBATCH --partition=cpuonly           # Queue for resource allocation
#SBATCH --time={wall_time}:00         # Wall-clock time limit
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node={n_tasks}   # Number of tasks per node
#SBATCH --cpus-per-task={n_trees//n_tasks}   # Number of tasks per node

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-special-couscous/}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export PYDIR=${{BASE_DIR}}/special-couscous/specialcouscous

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi
source "${{BASE_DIR}}"/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES={10**log_n_samples}
N_FEATURES={10**log_n_features}
N_TREES={n_trees}

SCRIPT="scripts/examples/rf_parallel_synthetic.py"

RESDIR="${{BASE_DIR}}"/results/acc_drop/n${{N_SAMPLES}}_m${{N_FEATURES}}/ntasks_${{SLURM_NPROCS}}/${{SLURM_JOB_ID}}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{PYDIR}}/${{SCRIPT}} \\
    --n_samples ${{N_SAMPLES}} \\
    --n_features ${{N_FEATURES}} \\
    --n_classes {n_classes}
    --random_state {random_state_data} \\
    --random_state_model {random_state_model} \\
    --n_trees ${{N_TREES}} \\
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
    n_trees = 76  # Number of trees in global forest
    n_tasks = [2, 4, 19, 38, 76]  # Considered number of tasks per node
    # Considered datasets given as pairs of (log10 <number of samples>, log10 <number of features>, wall-time):
    datasets = [(5, 3, 10), (6, 2, 30)]
    n_classes = 10
    # Considered seeds:
    seeds_data = [0, 1, 2, 3, 4]
    seeds_model = [5, 6, 7, 8, 9]
    output_path = pathlib.Path("dropping_acc/")

    for random_state_data in seeds_data:  # Loop over five different dataset seeds.
        for (
            random_state_model
        ) in seeds_model:  # Test five model seeds for each dataset seed.
            for dataset in datasets:  # Loop over considered datasets.
                log_n_samples = dataset[0]  # Extract number of samples.
                log_n_features = dataset[1]  # Extract number of features.
                wall_time = dataset[2]
                for num_tasks in n_tasks:  # Loop over different numbers of tasks.
                    generate_parallel_acc_drop_job_scripts(
                        log_n_samples=log_n_samples,
                        log_n_features=log_n_features,
                        wall_time=wall_time,
                        n_classes=n_classes,
                        n_trees=n_trees,
                        n_tasks=num_tasks,
                        random_state_data=random_state_data,
                        random_state_model=random_state_model,
                        output_path=output_path,
                        submit=False,
                    )
