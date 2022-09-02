import sys
import numpy as np
import util



def test_baseline(data: dict, model) -> float:
    X = data["X"]
    y = data["y"]
    
    num_correct = 0
    for label in y.tolist():
        prediction = 0
        if label == prediction:
            num_correct += 1

    accuracy = num_correct / len(y)

    return accuracy

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/baselines.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # Load configs
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    lr = float(configs["learning_rate"])
    dataset_fn = configs["filepaths"]["dataset"]

    # Load data
    raw_data = util.load_category_data(fn=dataset_fn)

    acc = test_baseline(raw_data, model = None) # dummy model for now
    print("predict false accuracy: ", acc)
        
    print("Done!")   

if __name__ == "__main__":
    main()