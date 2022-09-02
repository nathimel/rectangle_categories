#!/bin/sh

# Example: ./scripts/run_full_experiment.sh configs/dev.yml

if test $# -lt 1
then
    echo "Usage: ./scripts/main_results/run.sh path_to_config"
    exit 1
fi

CONFIG=$1

echo "python3 generate_categories.py $1"
python3 generate_categories.py $1

echo "python3 train.py $1"
python3 train.py $1

echo "python3 analyze.py $1"
python3 analyze.py $1
