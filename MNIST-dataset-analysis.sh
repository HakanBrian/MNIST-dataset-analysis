#!/bin/bash
#SBATCH --time=1-00:00:00           # Time limit for the job (REQUIRED).
#SBATCH --job-name=MNIST-analysis   # Job name
#SBATCH --nodes=1                   # Number of nodes to allocate. Same as SBATCH -N (Don't use this option for mpi jobs)
#SBATCH --exclusive                 # Allocate all cores in node.
#SBATCH --partition=short           # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm-%j.err             # Error file for this job.
#SBATCH -o slurm-%j.out             # Output file for this job.
#SBATCH -A <your project account>   # Project allocation account name (REQUIRED)
#SBATCH --mail-type ALL             # Send email when job starts/ends
#SBATCH --mail-user <your email>    # Email address to send email to

module load Miniconda3
# source /pathtoconda
# conda activate /pathtoenv/python_env

python MNIST-dataset-analysis.py