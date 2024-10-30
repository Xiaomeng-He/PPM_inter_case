import pandas as pd
import numpy as np

def get_train_test_split_point(df, test_ratio, 
                               case_id = 'case:concept:name', timestamp = 'time:timestamp'):
    """
    Get the splitting point of train / test split.

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    test_ratio: float
        The percentage of the test set
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp

    Returns
    -------
    train_test_split_time: pandas.Datetime
        The timestamp of the first event in test set
    train_test_split_idx: int
        The index of the first event in test set
    """
    # Get the start time of each case and sort by start time
    case_start_times = df.groupby(case_id)[timestamp].min().sort_values() # The results is a series
   
    # Get the index of the first case in test set
    first_test_case_idx = int(len(case_start_times) * (1 - test_ratio)) 
    
    # Get the case_id of the first case in test set
    first_test_case_id = case_start_times.index[first_test_case_idx]
    
    # Get the index of the first event of the first case in test set
    train_test_split_idx = df[df[case_id] == first_test_case_id].index[0]
    
    # Get the start time of the first case in test set, which is the split point of training set and test set
    train_test_split_time = df.loc[train_test_split_idx, timestamp]
    
    return train_test_split_time, train_test_split_idx

def get_discard_case_list(df, test_ratio, 
                          case_id = 'case:concept:name', timestamp = 'time:timestamp'):
    """
    Get the list of cases that should be removed when generating prefix-suffix for training set, i.e. cases that start before the splitting point and end after it.

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    test_ratio: float
        The percentage of the test set
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp

    Returns
    -------
    discard_case_list: list
        List of case ID of cases being diacarded
    """
    # Get the start time and end time of each case, then create a dataframe
    case_start_times = df.groupby(case_id)[timestamp].min().rename('Start') # The result is a series
    case_stop_times = df.groupby(case_id)[timestamp].max().rename('End')
    case_start_stop_times = pd.concat([case_start_times, case_stop_times], axis=1).reset_index() # The result is a dataframe
    train_test_split_time, _ = get_train_test_split_point(df, test_ratio, case_id, timestamp)

    # Get the list of cases that start before the spliiting point and end after the splitting point
    discard_cases = case_start_stop_times[
        (case_start_stop_times['Start'] < train_test_split_time) & 
        (case_start_stop_times['End'] > train_test_split_time)]
    discard_case_list = discard_cases[case_id].tolist()
    
    return discard_case_list

def create_table_without_discard_case(df, test_ratio,
                                      case_id = 'case:concept:name', timestamp = 'time:timestamp'):
    """
    Create table for training

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    test_ratio: float
        The percentage of the test set
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp

    Returns
    -------
    df_without_discard_case: pandas.DataFrame
        Event log

    """
    discard_case_list = get_discard_case_list(df, test_ratio, case_id, timestamp)
    df_without_discard_case = df[~df[case_id].isin(discard_case_list)]

    return df_without_discard_case

