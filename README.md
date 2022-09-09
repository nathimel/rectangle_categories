# Rectangle Categories

This repo is home for a pilot experiment for measuring whether simple NNs track the cognitive complexity of 2D categories.

## Overview

Steps of the experiment

1. Construct the 12 rectangle categories from Fass and Feldman (2002).
2. Construct a dataset from these categories.
3. Initialize a small population of neural learners.
4. Train each learner to learn each category, and report its learning effort (as e.g., average loss over epochs) as a measure of cognitive complexity.
5. Compare (e.g., linear regression) the learners' cognitive complexity data to the MDL complexity specified by Fass and Feldman (2002).

## Running

Reproduce the experiment by running `./run_full_experiment.sh path_to_config_file`

This calls the following scripts which can be run individually:

`python3 src/create_folders.py path_to_config_file`

`python3 src/generate_category_data.py path_to_config_file`

`python3 src/sample_train.py path_to_config_file`

`python3 src/main_experiment.py path_to_config_file`

`python3 src/analyze.py path_to_config_file`

A YAML configuration file should be created in [/configs](/configs) specifying parameters of the experiment (such as NN architecture, etc).
