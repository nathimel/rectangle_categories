"""Specify by hand the 12 categories of the experiment."""
import numpy as np

# A concept is any four-element subset of the model.
# A hypothesis is any set of rectangle classes that capture the concept.

categories = {
    '1': {
        'concept': np.array([ # input to each learner
            [0, 0, 0, 0,],
            [0, 0, 0, 0,],
            [1, 1, 0, 0,],
            [1, 1, 0, 0,],
            ]), 
        'hypothesis': np.array([ # not used directly by experiment, but may be useful
            [0, 0, 0, 0,],
            [0, 0, 0, 0,],
            [1, 1, 0, 0,],
            [1, 1, 0, 0,],
            ]), 
        'codelength': 8.0768, # if NNs track cognitive complexity, loss should be monotonically increasing with concept codelengths.
    },
    '2': {
        'concept': np.array([
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            ]),
        'hypothesis': np.array([
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            ]),
        'codelength': 8.3219,
    },
    '3': {
        'concept': np.array([
            [1, 0, 0, 0,],
            [0, 1, 0, 0,],
            [0, 0, 1, 0,],
            [0, 0, 0, 1,],
            ]),
        'hypothesis': np.array([
            [1, 1, 0, 0,],
            [1, 1, 0, 0,],
            [0, 0, 1, 1,],
            [0, 0, 1, 1,],
            ]),
        'codelength': 27.3236,
    },
    '4': {
        'concept': np.array([
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            [0, 0, 1, 0,],
            [0, 0, 1, 0,],
            ]),
        'hypothesis': np.array([
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            [0, 0, 1, 0,],
            [0, 0, 1, 0,],
            ]),
        'codelength': 17.8138,
    }, 
    '5': {
        'concept': np.array([
            [0, 0, 0, 0,],
            [1, 0, 1, 0,],
            [1, 0, 1, 0,],
            [0, 0, 0, 0,],
            ]), 
        'hypothesis': np.array([
            [0, 0, 0, 0,],
            [1, 1, 1, 0,],
            [1, 1, 1, 0,],
            [0, 0, 0, 0,],
            ]),
        'codelength': 16.5216,
    },
    '6': {
        'concept': np.array([
            [1, 1, 0, 0,],
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            [0, 0, 0, 0,],
            ]), 
        'hypothesis': np.array([
            [1, 1, 0, 0,],
            [1, 1, 0, 0,],
            [1, 1, 0, 0,],
            [0, 0, 0, 0,],
            ]),
        'codelength': 14.4919,
    },
    '7': {
        'concept': np.array([
            [0, 0, 1, 1,],
            [0, 0, 0, 0,],
            [1, 0, 0, 0,],
            [1, 0, 0, 0,],
            ]),
        'hypothesis': np.array([
            [1, 1, 1, 1,],
            [1, 1, 1, 1,],
            [1, 1, 1, 1,],
            [1, 1, 1, 1,],
            ]),
        'codelength': 17.1357,
    },
    '8': {
        'concept': np.array([
            [1, 0, 1, 0,],
            [0, 0, 0, 0,],
            [0, 1, 0, 0,],
            [0, 1, 0, 0,],
            ]), 
        'hypothesis': np.array([
            [1, 1, 1, 0,],
            [0, 0, 0, 0,],
            [0, 1, 0, 0,],
            [0, 1, 0, 0,],
            ]),
        'codelength': 22.5687,
    },
    '9': {
        'concept': np.array([
            [1, 0, 0, 0,],
            [1, 1, 0, 0,],
            [1, 0, 0, 0,],
            [0, 0, 0, 0,],
            ]), 
        'hypothesis': np.array([
            [1, 1, 0, 0,],
            [1, 1, 0, 0,],
            [1, 1, 0, 0,],
            [0, 0, 0, 0,],
            ]), 
        'codelength': 14.4919,
    },
    '10': {
        'concept': np.array([
            [1, 0, 0, 0,],
            [1, 0, 1, 0,],
            [1, 0, 0, 0,],
            [0, 0, 0, 0,],
        ]), 
        'hypothesis': np.array([
            [1, 1, 1, 0,],
            [1, 1, 1, 0,],
            [1, 1, 1, 0,],
            [0, 0, 0, 0,],
            ]), 
        'codelength': 15.0768,
    },
    '11': {
        'concept': np.array([
            [0, 0, 0, 0,],
            [0, 0, 0, 0,],
            [0, 1, 0, 1,],
            [1, 0, 1, 0,],
            ]), 
        'hypothesis': np.array([
            [0, 0, 0, 0,],
            [0, 0, 0, 0,],
            [1, 1, 1, 1,],
            [1, 1, 1, 1,],
            ]), 
        'codelength': 27.1946,
    },
    '12': {
        'concept': np.array([
            [0, 1, 0, 0,],
            [0, 0, 0, 1,],
            [0, 0, 1, 0,],
            [1, 0, 0, 0,],
            ]), 
        'hypothesis': np.array([
            [0, 1, 1, 1,],
            [0, 1, 1, 1,],
            [0, 1, 1, 1,],
            [1, 0, 0, 0,],
            ]), 
        'codelength': 27.1946,
    },
}