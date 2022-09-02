#!/bin/sh
source ~/miniforge3/etc/profile.d/conda.sh # Local
conda activate signet

# main command
time ./scripts/run_full_experiment.sh configs/dev.yml
