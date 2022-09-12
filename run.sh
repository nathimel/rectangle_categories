#!/bin/sh
source ~/miniforge3/etc/profile.d/conda.sh # Local
# source ~/anaconda3/etc/profile.d/conda.sh # Patas
conda activate rectangles

# dev
# time ./scripts/run_full_experiment.sh configs/dev.yml

# run multiple architectures

# time ./scripts/run_full_experiment.sh configs/mlp0.yml
time ./scripts/run_full_experiment.sh configs/mlp1.yml
# time ./scripts/run_full_experiment.sh configs/cnn0.yml
# time ./scripts/run_full_experiment.sh configs/cnn1.yml

# time ./scripts/run_full_experiment.sh configs/mlp2.yml
