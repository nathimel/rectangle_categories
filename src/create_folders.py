"""Create all directories for an experiment specified in the main config file recursively if they don't exist already."""

import sys
from util import load_configs, make_path


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/create_folers.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    config_fn = sys.argv[1]
    configs = load_configs(config_fn)

    for file_key in configs["filepaths"]:
        if file_key != "data":
            make_path(configs["filepaths"][file_key])


if __name__ == "__main__":
    main()