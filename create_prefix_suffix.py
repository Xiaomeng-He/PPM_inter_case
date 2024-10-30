import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from train_test_split import get_train_test_split_point
from preprocessing import sort_log, create_time_features, mapping_case_id, test_mapping_event_name, train_mapping_event_name, add_soc_eoc

def create_log_prefix_tensor(df, 
                             log_prefix_length, 
                             set_name,
                             test_ratio,
                             num_act,
                             col_name = ['concept:name', 'log_ts_pre'],
                             categorical_features = ['concept:name'],
                             case_id = 'case:concept:name', 
                             timestamp = 'time:timestamp', 
                             event_name = 'concept:name'):
    """
    
    Create log prefix. Can choose between create for training set or create for test set. 

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    log_prefix_length: integer
        Length of max log prefix
    set_name: str, 'train' or 'test'
        Indicate whether to create log prefix for training set or for test set
    test_ratio: float
        The percentage of test set
    num_act: integer
        Number of activity labels (including padding, SOC, EOC, unknown label)
    col_name: list, optional
        Name(s) of column(s) containing  features.
    categorical_features: list, optional
        Name(s) of column(s) containing categorical features.
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp
    event_name: str, optional
        Name of column containing activity label

    Returns
    -------
    log_prefix_cat_tensor: tensor
        shape: (num_obs, log_prefix_len, num_act + 1)

    """

    # for training set, the input is already a subset of event log, so use whole input dataframe to create tensor
    if set_name == "train":
        split_idx = 0
    # for test set, the input datafram is the whole event log, need to subset here
    else:
        _, train_test_split_idx = get_train_test_split_point(df, test_ratio, 
                                                        case_id=case_id, timestamp=timestamp)
        split_idx = train_test_split_idx  
    
    # create an empty list to store tensors of different col_name
    tensors_list = []

    for col in col_name:

        # create an empty list to store all predix tensors pertaining to one col_name
        log_prefix_list = []

        # distinguish between categorical features and continuous features.
        # for categorical features, the masking number is 0 
        # for continuous features, the masking number is -10000
        if col in categorical_features:
            masking_number = int(0)
        else:
            masking_number = float(-10000)

        for i in range(split_idx, len(df)):

            # will not generate prefix ending with EOC (but will generate prefix that ending with SOC)
            if df[event_name].iloc[i] != 3 :

                start_idx = max(0, i - log_prefix_length + 1)
                
                prefix = df[col].iloc[start_idx:i+1].tolist() # Prefix includes the current event

                masking = [masking_number] * max(0, log_prefix_length - len(prefix)) # make sure not to multiply a negative number

                # apply left masking
                prefix = masking + prefix
                log_prefix_list.append(prefix)

        # create tensor for each col_name    
        log_prefix_tensor = torch.tensor(log_prefix_list)

        if col in categorical_features:
            # for categorical features, one-hot encoding will be applied
            log_prefix_tensor = log_prefix_tensor.long()
            log_prefix_tensor = F.one_hot(log_prefix_tensor, num_classes=num_act)
            # log_prefix_tensor shape: (num_obs, prefix_len, num_act)
            log_prefix_tensor[:, :, 0] = 0 # To ensure that 0 padding will be encoded as all 0s
        else:
            log_prefix_tensor = log_prefix_tensor.float() 
            # log_prefix_tensor shape: (num_obs, prefix_len)
            log_prefix_tensor = log_prefix_tensor.unsqueeze(2) 
            # log_prefix_tensor shape: (num_obs, prefix_len, 1)

        # list of tensors for all col_name
        tensors_list.append(log_prefix_tensor)

    # concatenate list of tensors by the last dimension.
    log_prefix_cat_tensor = torch.cat(tensors_list, dim=2)

    return log_prefix_cat_tensor

def create_trace_prefix_tensor(df, 
                             trace_prefix_length, 
                             set_name,
                             test_ratio,
                             num_act,
                             col_name = ['concept:name', 'trace_ts_start', 'trace_ts_pre'],
                             categorical_features = ['concept:name'],
                             event_idx = 'event_idx',
                             case_id = 'case:concept:name', 
                             timestamp = 'time:timestamp', 
                             event_name = 'concept:name'):
    """
    
    Create trace prefix. Can choose between create for training set or create for test set. 

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    trace_prefix_length: integer
        Length of max trace prefix
    set_name: str, 'train' or 'test'
        Indicate whether to create log prefix for training set or for test set
    test_ratio: float
        The percentage of test set
    num_act: integer
        Number of activity labels (including padding, SOC, EOC, unknown label)
    col_name: list, optional
        Name(s) of column(s) containing  features.
    categorical_features: list, optional
        Name(s) of column(s) containing categorical features.
    event_idx: str, optional, default: 'event_idx'
        Idex of event    
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp
    event_name: str, optional
        Name of column containing activity label   

    Returns
    -------
    trace_prefix_cat_tensor: tensor
        shape: (num_obs, trace_prefix_len, num_act + 2)

    """

    # for training set, the input is already a subset of event log, so use whole input dataframe to create tensor
    if set_name == "train":
        split_idx = 0
    # for test set, the input datafram is the whole event log, need to subset here
    else:
        _, train_test_split_idx = get_train_test_split_point(df, test_ratio, 
                                                        case_id=case_id, timestamp=timestamp)
        split_idx = train_test_split_idx  
    
    # create an empty list to store tensors of different col_name
    tensors_list = []

    for col in col_name:

        # create an empty list to store all prefix tensors pertaining to one col_name
        trace_prefix_list = []

        # distinguish between categorical features and continuous features.
        # for categorical features, the masking number is 0 
        # for continuous features, the masking number is -10000
        if col in categorical_features:
            masking_number = int(0)
        else:
            masking_number = float(-10000)

        for i in range(split_idx, len(df)):
            # will not generate prefix that ends with EOC (but will generate prefix that ends with SOC)
            if df[event_name].iloc[i] != 3 :
                # get the event idex of the current event
                current_event_idx = df[event_idx].iloc[i]
                # filter the dataframe to contains rows with the same case ID as the current event
                filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)
                # the rows before the current event in the filtered dataframe will be the prefix
                prefix = filtered_df[filtered_df[event_idx] <= current_event_idx][col].tolist() # Prefix includes the current event
                # restrict the length to trace_prefix_length
                if len(prefix) > trace_prefix_length:
                    prefix = prefix[-trace_prefix_length:]
                masking = [masking_number] * max(0, trace_prefix_length - len(prefix)) # make sure not to multiply a negative number
                # apply left masking
                prefix = masking + prefix
                trace_prefix_list.append(prefix)

        # create tensor for each col_name    
        trace_prefix_tensor = torch.tensor(trace_prefix_list)

        if col in categorical_features:
            # for categorical features, one-hot encoding will be applied
            trace_prefix_tensor = trace_prefix_tensor.long()
            trace_prefix_tensor = F.one_hot(trace_prefix_tensor, num_classes=num_act)
            trace_prefix_tensor[:, :, 0] = 0
        else:
            trace_prefix_tensor = trace_prefix_tensor.float()
            trace_prefix_tensor = trace_prefix_tensor.unsqueeze(2)

        # list of tensors for all col_name
        tensors_list.append(trace_prefix_tensor)

    # concatenate list of tensors by the last dimension.
    trace_prefix_cat_tensor = torch.cat(tensors_list, dim=2)

    return trace_prefix_cat_tensor

def create_trace_suffix_tensor(df, 
                             trace_suffix_length, 
                             set_name,
                             test_ratio,
                             col_name = ['concept:name', 'trace_ts_pre'],
                             categorical_features = ['concept:name'],
                             event_idx = 'event_idx',
                             case_id = 'case:concept:name', 
                             timestamp = 'time:timestamp', 
                             event_name = 'concept:name'):
    """
    
    Create trace prefix. Can choose between create for training set or create for test set. 

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    trace_suffix_length: integer
        Length of max trace suffix
    set_name: str, 'train' or 'test'
        Indicate whether to create log prefix for training set or for test set
    test_ratio: float
        The percentage of test set
    num_act: integer
        Number of activity labels (including padding, SOC, EOC, unknown label)
    col_name: list, optional
        Name(s) of column(s) containing  features.
    categorical_features: list, optional
        Name(s) of column(s) containing categorical features.
    event_idx: str, optional, default: 'event_idx'
        Idex of event    
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp
    event_name: str, optional
        Name of column containing activity label  

    Returns
    -------
    suffix_tensors_list: list
        shape of each tensor in the list: (num_obs, trace_suffix_len)

    """

    # for training set, the input is already a subset of event log, so use whole input dataframe to create tensor
    if set_name == "train":
        split_idx = 0
    # for test set, the input datafram is the whole event log, need to subset here
    else:
        _, train_test_split_idx = get_train_test_split_point(df, test_ratio, 
                                                        case_id=case_id, timestamp=timestamp)
        split_idx = train_test_split_idx
    
    # create an empty list to store tensors of different col_name
    suffix_tensors_list = []

    for col in col_name:

        # create an empty list to store all suffix tensors pertaining to one col_name
        trace_suffix_list = []

        # distinguish between categorical features and continuous features.
        # for categorical features, the masking number is 0 
        # for continuous features, the masking number is -10000
        if col in categorical_features:
            masking_number = int(0)
        else:
            masking_number = float(-10000)

        for i in range(split_idx, len(df)):
            # will not generate prefix that ends with EOC (but will generate prefix that ends with SOC)
            if df[event_name].iloc[i] != 3 :
                # get the event idex of the current event
                current_event_idx = df[event_idx].iloc[i]
                # filter the dataframe to contains rows with the same case ID as the current event
                filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)
                # the rows after the current event in the filtered dataframe will be the suffix
                suffix = filtered_df[filtered_df[event_idx] > current_event_idx][col].tolist()
                # restrict the length to trace_suffix_length
                if len(suffix) > trace_suffix_length:
                    suffix = suffix[:trace_suffix_length]
                masking = [masking_number] * max(0, trace_suffix_length - len(suffix)) # make sure that (trace_suffix_length - len(suffix)) would not be a negative number
                # apply right masking
                suffix = suffix + masking
                trace_suffix_list.append(suffix)
        # create tensor for each col_name    
        trace_suffix_tensor = torch.tensor(trace_suffix_list)

        # for categorical features, the type of the tensor should be long (for softmax)
        if col in categorical_features:
            trace_suffix_tensor = trace_suffix_tensor.long() # this is a 2D tensor (num_bs * seq_length)
        else:
            trace_suffix_tensor = trace_suffix_tensor.float()  # this is a 2D tensor (num_bs * seq_length)

        # list of tensors for all col_name
        suffix_tensors_list.append(trace_suffix_tensor)

    return suffix_tensors_list