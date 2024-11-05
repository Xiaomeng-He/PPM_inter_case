# Seq2Seq Model Combining Log Prefix and Trace Prefix for Predictive Process Monitoring
## Files Overview
`run_experiment_6000cases.ipynb`<br />
This notebook contains the complete workflow for running the experiment, from data preprocessing and model training to evaluation. Functions from other `.py` files are untilized in this notebook.

`preprocessing.py` <br />
Contains functions for preprocessing the event log in a tabular format.

`create_prefix_suffix.py` <br />
Contains functions that transform the preprocessed tables into prefix and suffix tensors.

`train_test_split.py` <br />
Contains functions to obtain the train/test split point, and create training tables that exclude cases spanning the split (i.e. cases starting before the splitting point and ending afterwards).

`dataloader_pipeline.py` <br />
Defines the complete pipeline for converting a raw CSV file into prefix and suffix tensors that could be directly fed into the model. By utilizing functions from `preprocessing.py`, `train_test_split.py` and `create_prefix_suffix.py`, two pipelines are built:<br /> 
1. A pipeline that takes CSV file as input and outputs train dataloader and validation dataloader;
2. A pipeline that takes CSV file as input and outputs test dataloader. 

In the first and second sections of `run_experiment_6000cases.ipynb`, the steps from `dataloader_pipeline.py` are mirrored and executed separately.

`create_model.py` <br />
Contains classes that define the encoder, decoder and seq2seq model, as well as a function to calculate the normalized Damerau-Levenshtein distance.

`BPIC2017_6000cases.csv`<br />
A sample of 6000 cases from the BPIC2017 dataset used for this experiment.
