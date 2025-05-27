#!/bin/bash
#PBS -l walltime=00:01:00
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -N test_script
#PBS -j oe

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Initialize Conda (specific to Miniforge3)
source ~/miniforge3/etc/profile.d/conda.sh

# Activate your environment
conda activate test_env

# Run your job command
python test_script.py