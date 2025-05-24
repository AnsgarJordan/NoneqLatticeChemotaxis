#!/bin/bash
#PBS -l walltime=00:01:00
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -N test_script
#PBS -j oe

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Load the Conda module if needed (some systems require it to initialize conda)
module load anaconda/2.4.1  # or whichever Anaconda module is available

# Activate your environment (assumes it's in your home directory or custom path)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env

# Run your job command
python test_script.py