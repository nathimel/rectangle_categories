#!/bin/sh
source ~/miniforge3/etc/profile.d/conda.sh # Local
conda activate rectangles

# dev
time ./scripts/run_full_experiment.sh configs/dev.yml

# run multiple architectures

# time ./scripts/run_full_experiment.sh configs/mlp0.yml
# time ./scripts/run_full_experiment.sh configs/mlp1.yml
# time ./scripts/run_full_experiment.sh configs/cnn0.yml
# time ./scripts/run_full_experiment.sh configs/cnn1.yml
