import subprocess

nodes = [2, 4, 8, 16, 32, 64]  # Number of nodes for scaling exps
num_trees_base = 800


def main() -> None:
    """Generate the n6_m4 job scripts."""
    for num_nodes in nodes:
        num_trees = num_trees_base * num_nodes
        print(f"Current config uses {num_nodes} nodes and {num_trees} trees.")
        job_name = f"n6_m4_weak_{num_nodes}"
        job_script_name = f"{job_name}.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}         # Job name
#SBATCH --partition=cpuonly           # Queue for resource allocation
#SBATCH --time=2-12:00:00             # Wall-clock time limit
#SBATCH --mem=243200mb		          # Main memory (use standard nodes)
#SBATCH --cpus-per-task=76            # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --nodes={num_nodes}           # Number of nodes
#SBATCH --ntasks-per-node=1           # One MPI rank per node

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-special-couscous/}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export PYDIR=${{BASE_DIR}}/special-couscous/specialcouscous

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi
source "${{BASE_DIR}}"/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES=1000000
N_FEATURES=10000
N_TREES={num_trees}

SCRIPT="rf_parallel_synthetic.py"

RESDIR="${{BASE_DIR}}"/results/weak_scaling/n6_m4_nodes_${{SLURM_NPROCS}}_${{SLURM_JOB_ID}}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{PYDIR}}/${{SCRIPT}} --n_samples ${{N_SAMPLES}} --n_features ${{N_FEATURES}} --n_trees ${{N_TREES}} --output_dir ${{RESDIR}} --output_label ${{SLURM_JOB_ID}} --detailed_evaluation --save_model
                                """

        with open(job_script_name, "wt") as f:
            f.write(script_content)

        subprocess.run(f"sbatch {job_script_name}", shell=True)


if __name__ == "__main__":
    main()
