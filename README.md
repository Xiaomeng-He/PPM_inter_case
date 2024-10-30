# Seq2Seq Model Combining Log Prefix and Trace Prefix for Preditive Process Monitoring
## File Structure
`run_experiment_6000cases.ipynb` comprises all steps to run the experiment and displays results based a sample of 6000 cases from BPIC2017. Functions from other `.py` files are used in this main notebook. `BPIC2017_6000cases.csv` is used for this run of experiment.

`preprocessing.py` includes functions used for preprocessing.

`train_test_split.py` includes functions to get the train / test splitting point and to create tables without discard cases (i.e. cases starting before the splitting point and ending afterwards).

`create_prefix_suffix.py` includes functions that take preprocessed dataframe as input, and output prefix tensors or suffix tensors.

`dataloader_pipeline.py` includes two pipelines: one takes csv file as input and outputs train dataloader and validation dataloader; the other takes csv file as input and outputs test dataloader. The steps of these pipelines are mirrored and executed separately in `run_experiment_6000cases.ipynb`.

`create_model.py` includes classes that define encoder, decoder and seq2seq model, along with a function to calculate normalized Damerau-Levenshtein Distance.
