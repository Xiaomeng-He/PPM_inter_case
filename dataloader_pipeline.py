import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader
from preprocessing import sort_log, debiasing, create_time_features, mapping_case_id, add_soc_eoc
from preprocessing import train_mapping_event_name, test_mapping_event_name, train_standardize, test_standardize
from train_test_split import create_table_without_discard_case, get_train_test_split_point
from create_prefix_suffix import create_log_prefix_tensor, create_trace_prefix_tensor, create_trace_suffix_tensor

def create_train_valid_dataloader(csv_path, end_date, max_duration,
                                  test_ratio, validation_ratio, num_act,
                                  log_prefix_length, trace_prefix_length, trace_suffix_length, 
                                  set_name = 'train', 
                                  log_col_name = ['concept:name', 'log_ts_pre'],
                                  trace_prefix_col_name = ['concept:name', 'trace_ts_pre', 'trace_ts_start'],
                                  trace_suffix_col_name = ['concept:name', 'trace_ts_pre'],
                                  categorical_features = ['concept:name'],
                                  continuous_features = ['log_ts_pre', 'trace_ts_pre', 'trace_ts_start'],
                                  case_id = 'case:concept:name', 
                                  timestamp = 'time:timestamp', 
                                  event_name = 'concept:name',
                                  event_idx = 'event_idx'):
    """
    Transform event logs from csv file to training dataloader and validation dataloader.

    Parameters
    ----------
    csv_path: str
        The path to csv file containing event log
    test_ratio: float
        The percentage of test data
    validation_ratio: float
        The percentage of validation set
    log_prefix_length: int
        length of log prefix
    trace_prefix_length: int
        length of trace prefix
    trace_suffix_length: int
        length of trace suffix

    Returns
    -------
    train_dataloader: DataLoader
        Dataloader for training set
    valid_dataloader: DataLoader
        Dataloader for validation set
    train_event_name_dict: dictionary
        Map event_name to numerical index
    mean_dict: dictionary
        The keys store column name and the values store corresponding mean.
    std_dict: dictionary
        The keys store column name and the values store corresponding standard deviation.
    """
    # 1. Tranform csv to dataframe
    df = pd.read_csv(csv_path)

    # 2. Sort dataframe by timestamp -> preprocessing.sort_log
    df = sort_log(df,
                  timestamp)
    
    # 3. Debiasing and cleaning -> preprocessing.debiasing
    df = debiasing(df, end_date, max_duration,
                   case_id, timestamp)
    
    # 4. Get rid of discard case -> train_test_split.create_table_without_discard_case
    df_no_discard = create_table_without_discard_case(df, test_ratio, 
                                           case_id, timestamp)
    
    # 5. Subset: retain dataframe only before training / test split
    # Note: the argument for get_train_test_split_point is df rather than df_no_discard
    train_test_split_time, train_test_split_idx = get_train_test_split_point(df, test_ratio, 
                                                          case_id, timestamp)
    
    training_df = df_no_discard[df_no_discard[timestamp] < train_test_split_time]

    # 6. Create time features
    training_df = create_time_features(training_df, 
                                       case_id,timestamp)
    
    # 7. Standardize time features
    training_df, mean_dict, std_dict = train_standardize(training_df, 
                                                         continuous_features)
    
    # 8. Map case ID to numbers
    training_df, case_id_dict = mapping_case_id(training_df, 
                                     case_id)
    
    # 9. Insert SOC and EOC rows
    training_df = add_soc_eoc(training_df, 
                              case_id, timestamp, event_name)
    
    # 10. Mapping event name to numbers
    training_df, train_event_name_dict = train_mapping_event_name(training_df, 
                                                                  event_name)
    
    # 11. Create train_log_prefix_tensor, train_trace_prefix_tensor, train_suffix_act_tensor, train_suffix_time_tensor
    train_log_prefix_tensor = create_log_prefix_tensor(training_df, log_prefix_length, set_name, test_ratio, num_act,
                                                       log_col_name, categorical_features, case_id, timestamp, event_name)
    train_trace_prefix_tensor = create_trace_prefix_tensor(training_df, trace_prefix_length, set_name, test_ratio, num_act,
                                                           trace_prefix_col_name, categorical_features, event_idx, case_id, timestamp, event_name)
    train_suffix_act_tensor, train_suffix_time_tensor = create_trace_suffix_tensor(training_df, trace_suffix_length, set_name, test_ratio,
                                                                                   trace_suffix_col_name, categorical_features, event_idx, case_id, timestamp, event_name)
    
    # 12. Build TensorDataset
    train_dataset = TensorDataset(train_log_prefix_tensor, train_trace_prefix_tensor, train_suffix_act_tensor, train_suffix_time_tensor)

    # 13. Split TensorDataset into train_dataset, valid_dataset
    train_dataset, valid_dataset = random_split(train_dataset, [1-validation_ratio, validation_ratio])

    # 14. Build train_dataloader, valid_dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

    return train_dataloader, valid_dataloader, train_event_name_dict, mean_dict, std_dict

def create_test_dataloader(csv_path, end_date, max_duration,
                           test_ratio, num_act,
                           log_prefix_length, trace_prefix_length, trace_suffix_length, 
                           train_event_name_dict, mean_dict, std_dict,
                           set_name = 'test', 
                           log_col_name = ['concept:name', 'log_ts_pre'],
                           trace_prefix_col_name = ['concept:name', 'trace_ts_pre', 'trace_ts_start'],
                           trace_suffix_col_name = ['concept:name', 'trace_ts_pre'],                            
                           categorical_features = ['concept:name'],
                           continuous_features = ['log_ts_pre', 'trace_ts_pre', 'trace_ts_start'],
                           case_id = 'case:concept:name', 
                           timestamp = 'time:timestamp', 
                           event_name = 'concept:name',
                           event_idx = 'event_idx'):
    """
    Transform event logs from csv file to test dataloader.

    Parameters
    ----------
    csv_path: str
        The path to csv file containing event log
    test_ratio: float
        The percentage of test data
    log_prefix_length: int
        length of log prefix
    trace_prefix_length: int
        length of trace prefix
    trace_suffix_length: int
        length of trace suffix
    num_act: int
        number of 
    train_event_name_dict: dictionary
        Map event_name to numerical index
    mean_dict: dictionary
        The keys store column name and the values store corresponding mean.
    std_dict: dictionary
        The keys store column name and the values store corresponding standard deviation.   

    Returns
    -------
    test_dataloader: DataLoader
        Dataloader for test set

    """
    # 1. Tranform csv to dataframe
    df = pd.read_csv(csv_path)

    # 2. Sort dataframe by timestamp -> preprocessing.sort_log
    df = sort_log(df,
                  timestamp)
    
    # 3. Debiasing and cleaning -> preprocessing.debiasing
    df = debiasing(df, end_date, max_duration,
                   case_id, timestamp)

    # 4. Create time features
    df = create_time_features(df,
                              case_id,timestamp)

    # 5. Standardize time features
    df = test_standardize(df,
                          mean_dict, std_dict, continuous_features)
    
    # 6. Map case ID to numbers
    df, case_id_dict = mapping_case_id(df, 
                            case_id)
    
    # 7. Insert SOC and EOC rows
    df = add_soc_eoc(df,
                     case_id, timestamp, event_name)

    # 8. Mapping event name to numbers
    df, test_event_name_dict = test_mapping_event_name(df, train_event_name_dict, 
                                                       event_name)
    
    # 9. Create test_log_prefix_tensor, test_trace_prefix_tensor, test_suffix_act_tensor, test_suffix_time_tensor
    test_log_prefix_tensor = create_log_prefix_tensor(df, log_prefix_length, set_name, test_ratio, num_act,
                                                       log_col_name, categorical_features, case_id, timestamp, event_name)
    test_trace_prefix_tensor = create_trace_prefix_tensor(df, trace_prefix_length, set_name, test_ratio, num_act,
                                                           trace_prefix_col_name, categorical_features, event_idx, case_id, timestamp, event_name)
    test_suffix_act_tensor, test_suffix_time_tensor = create_trace_suffix_tensor(df, trace_suffix_length, set_name, test_ratio,
                                                                                   trace_suffix_col_name, categorical_features, event_idx, case_id, timestamp, event_name)
    
    # 10. Build TensorDataset
    test_dataset = TensorDataset(test_log_prefix_tensor, test_trace_prefix_tensor, test_suffix_act_tensor, test_suffix_time_tensor)

    # 11. Build test_dataloader, valid_dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_dataloader

