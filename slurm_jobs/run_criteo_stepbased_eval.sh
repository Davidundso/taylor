#!/bin/bash

# Slurm job script for executing the experiment

#SBATCH -J criteo1tb_train          # Job name
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --nodes=1                   # Ensure that all cores are on the same machine (single node)
#SBATCH --partition=a100-galvani    # Partition to run your job on (adjust if necessary)
#SBATCH --time=3-00:00              # Allowed runtime in D-HH:MM (adjust if needed)
#SBATCH --gres=gpu:1                # Requesting GPUs (if needed)
#SBATCH --mem=120G                   # Total memory pool for all cores (adjust if necessary)
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

# Or if using conda:
source ~/.bashrc
conda activate $WORK/.conda/algo2

# Compute Phase - running the experiment
python3 submission_runner_stepbased_eval.py \
    --framework=pytorch \
    --workload=criteo1tb \
    --data_dir=/mnt/lustre/datasets/mlcommons/criteo1tb \
    --tuning_ruleset=self \
    --experiment_dir=$WORK/cluster_experiments \
    --experiment_name=criteo_stepb_eval0512 \
    --submission_path=my_submissions/sub2/sub2_criteo_debugged_timing.py \
    --torch_compile=False \
    --max_global_steps=10000

conda deactivate
