# I3SP

## Introduction

This repository contains the code and implementation details for the paper **Inter-case Informed Business Process Suffix Prediction Integrating Trace and Log Information**.

To facilitate navigation, below is an overview of the files:

- **`1_create_prefix_suffix/`**  
  Contains scripts for data preprocessing, dataset splitting, and generating trace prefixes, log prefixes, and trace suffixes:  
  - `preprocessing.py`: Handles data preprocessing
  - `train_test_split.py`: Handles dataset splitting.  
  - `create_prefix_suffix.py`: Generates trace prefixes, log prefixes, and trace suffixes.  

- **`2_Seq2Seq/`**  
  Contains scripts for creating both trace-based and integrated Seq2Seq models:  
  - `create_Seq2Seq.py`: Implements the Seq2Seq model.  

- **`3_SEP_LSTM/`**  
  Contains scripts for creating both trace-based and integrated SEP-LSTM models:  
  - `create_SEP_LSTM.py`: Implements the SEP-LSTM model.  

- **`next_event_prediction_metrics.md`**  
  Documents the performance metrics of SEP-LSTM and SEP-XGBoost for two supplementary prediction tasks:  
  - Next activity label prediction
  - Next timestamp prediction

## Data Preprocessing

### Data Clearning and Debiasing

### Dataset Spliting

### Sample Generation

## Seq2Seq Model

## SEP-LSTM

## SEP-XGBoost
