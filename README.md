# krakencoder

## CLI-facing scripts:
* [`run_training.py`](run_training.py): Train a new model
* [`run_model.py`](run_model.py): Run a saved checkpoint on new data
* [`describe_checkpoint.py`](describe_checkpoint.py): Print information about a saved checkpoint

## Internal scripts:
* [`model.py`](krakencoder/model.py): Krakencoder model class definition
* [`train.py`](krakencoder/train.py): Training-related functions
* [`loss.py`](krakencoder/loss.py): Specifies different loss functions to be used during training and evaluation
* [`data.py`](krakencoder/data.py): Functions for loading and transforming input data
* [`plotfigures.py`](krakencoder/plotfigures.py): Functions for plotting loss curves and performance heatmaps
* [`utils.py`](krakencoder/utils.py): Miscellaneous utility functions