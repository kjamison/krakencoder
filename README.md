# krakencoder

## CLI-facing scripts:
* [`run_training.py`](run_training.py): Train a new model
* [`run_model.py`](run_model.py): Run a saved checkpoint on new data
* [`describe_model.py`](describe_model.py): Print information about a saved checkpoint

## Internal scripts:
* [`krakencoder.py`](krakencoder.py): Model class definition
* [`train.py`](train.py): Training-related functions
* [`loss.py`](loss.py): Specifies different loss functions to be used during training and evaluation
* [`data.py`](data.py): Functions for loading and transforming input data
* [`utils.py`](utils.py): Miscellaneous utility functions
* [`plotfigures.py`](plotfigures.py): Functions for plotting loss curves and performance heatmaps