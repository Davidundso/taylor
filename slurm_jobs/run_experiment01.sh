#!/bin/bash

# Slurm job script for executing the experiment

#SBATCH -J cluster_exp01            # Job name
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --nodes=1                   # Ensure that all cores are on the same machine (single node)
#SBATCH --partition=2080-galvani    # Partition to run your job on (adjust if necessary)
#SBATCH --time=0-12:00             # Allowed runtime in D-HH:MM (adjust if needed)
#SBATCH --gres=gpu:2                # Requesting GPUs (if needed)
#SBATCH --mem=50G                   # Total memory pool for all cores (adjust if necessary)
#SBATCH --output=$WORK/cluster_experiments/cluster_exp01/job-%j.out   # STDOUT file
#SBATCH --error=$WORK/cluster_experiments/cluster_exp01/job-%j.err    # STDERR file
#SBATCH --mail-type=ALL             # Email notifications on job status
#SBATCH --mail-user=david.suckrow@student.uni-tuebingen.de  # Your email for notifications

# Diagnostic Phase - shows job info and system state
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi                          # Display GPU info (only if you requested GPUs)
ls $WORK                             # Just to confirm $WORK directory is accessible

# Setup Phase - activate virtual environment if needed
cd $HOME/taylor/algorithmic_efficiency  # Change to your project directory
# If you're using a virtual environment:
# source <path_to_virtualenv>/bin/activate
# Or if using conda:
conda activate $WORK/.conda/algo2

# Compute Phase - running the experiment
python3 submission_runner.py \
    --framework=pytorch \
    --workload=fastmri \
    --workload_path=/mnt/lustre/datasets/mlcommons/fastmri \
    --experiment_dir=$WORK/cluster_experiments/cluster_exp01 \
    --experiment_name=cluster_exp01 \
    --submission_path=my_submissions/sub2/sub2_baseline_alpha.py
