import pandas as pd

def sort_log(df, 
             timestamp = 'time:timestamp'):
    """    
    Transform the format of timestamp and sort the event logs by timestamp

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    timestamp: str, optional
        Name of column containing timestamp
                
    Returns
    -------
    sorted_df: pandas.DataFrame
        Event log sorted by timestamp

    """
    # convert timestamp to pandas datetime
    df[timestamp] = pd.to_datetime(df['time:timestamp'], format='mixed').dt.tz_convert('UTC')

    # sort event log by timestamp
    sorted_df = df.sort_values(by=timestamp).reset_index(drop=True)

    return sorted_df

def debiasing(df,
              end_date,
              max_duration,
              case_id = 'case:concept:name', 
              timestamp = 'time:timestamp'):
    """
    Remove chronological outliers and debiasing the end of the dataset.

    The code and the setting of parameters (end_date, max_duration) is adapted from the follwing paper:
    Weytjens, H., De Weerdt, J. (2022). Creating Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive Process Monitoring. In: Marrella, A., Weber, B. (eds) Business Process Management Workshops. BPM 2021. Lecture Notes in Business Information Processing, vol 436. Springer, Cham. 

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    end_date: str
        Cases ending after this month will be removed
    max_duration: float
        Maximum days a normal case lasts
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp

    Returns
    -------
    df: pandas.Dataframe
        Debiased event log  

    """
    # remove outliers ending after end date
    case_stops_df = pd.DataFrame(df.groupby(case_id)[timestamp].max().reset_index())
    case_stops_df['date'] = case_stops_df[timestamp].dt.to_period('M')
    cases_before = case_stops_df[case_stops_df['date'].astype('str') <= end_date][case_id].values
    df = df[df[case_id].isin(cases_before)]

    # compute each case's duration
    agg_dict = {timestamp :['min', 'max']}
    duration_df = pd.DataFrame(df.groupby(case_id).agg(agg_dict)).reset_index()
    duration_df["duration"] = (duration_df[(timestamp,"max")] - duration_df[(timestamp,"min")]).dt.total_seconds() / (24 * 60 * 60)

    # retain only cases shorter than max duration
    condition_1 = duration_df["duration"] <= max_duration *1.00000000001
    cases_retained = duration_df[condition_1][case_id].values
    df = df[df[case_id].isin(cases_retained)].reset_index(drop=True)

    # drop cases starting after the dataset's last timestamp minus the max_duration
    latest_start = df[timestamp].max() - pd.Timedelta(max_duration, unit='D')
    condition_2 = duration_df[(timestamp, "min")] <= latest_start
    cases_retained = duration_df[condition_2][case_id].values
    df = df[df[case_id].isin(cases_retained)].reset_index(drop=True)

    # ensure event log is sorted by timestamp
    df = df.sort_values(by=timestamp).reset_index(drop=True)
    
    return df 

def create_time_features(df, 
                        case_id = 'case:concept:name', timestamp = 'time:timestamp'):
    """
    
    Create three new time fetures: 
    - log_ts_pre: time since the previou event in the event log
    - trace_ts_pre: time since the previous event in the case
    - trace_ts_statr: time since the start of the case (i.e. the first event in this case)
    
    Parameters
    ----------
    df: pandas.DataFrame
        Event log 
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with three new time features
    
    """
    # ensure the event log is sorted by timestamp
    df = df.sort_values(by=timestamp).reset_index(drop=True)

    # calulate time since the previous event in event log
    df['log_ts_pre'] = df[timestamp].diff(periods=1).dt.total_seconds()
    df.loc[0, 'log_ts_pre'] = 0.0

    # calculate time since the previous event in case
    df = df.sort_values(by=[case_id, timestamp]).reset_index(drop=True)
    df['trace_ts_pre'] = 0.0
    for i in range(1, len(df)): # start from 1 to avoid calculating i-1 when i=0
        if df[case_id].iloc[i] == df[case_id].iloc[i - 1]: # if it is not the first event in each case
            df.loc[i, 'trace_ts_pre'] = (df[timestamp].iloc[i] - df[timestamp].iloc[i - 1]).total_seconds()

    # a helper column containing the start time of each case
    case_start_times_df = df.groupby(case_id)[timestamp].min().to_frame(name='case_start_time').reset_index(names=case_id) 
    df = pd.merge(df, case_start_times_df, on=case_id, how='inner')

    # calculte time since the first event in case
    df['trace_ts_start'] = (df[timestamp] - df['case_start_time']).dt.total_seconds()

    # drop the helper column
    df = df.drop(columns=['case_start_time'])
    df = df.sort_values(by=timestamp).reset_index(drop=True)

    return df

def train_standardize(df, 
                     col_name=['log_ts_pre', 'trace_ts_pre', 'trace_ts_start']):
    """
    Standardize continuous features in training set.

    Parameters
    ----------
    df: pandas.Dataframe
        Event log
    col_name: list, optional
        Name(s) of column(s) containing continuous features.
    
    Returns
    -------
    df: pandas.Dataframe
        Dataframe with standardized continuous features.
    mean_dict:dictionary
        The keys are feature names and the values are mean values.
    std_dict: dictionary
        The keys are feature names and the values are standard deviation values.
    """
    # initialize dictionaries to store mean and std for each feature
    mean_dict = {}
    std_dict = {}

    # loop through all features
    for col in col_name:
        mean = df[col].mean()
        std = df[col].std()
        
        # store the mean and std in the dictionaries
        mean_dict[col] = mean
        std_dict[col] = std
        
        # standardize the column
        df[col] = (df[col] - mean) / std
    
    return df, mean_dict, std_dict

def test_standardize(df, mean_dict, std_dict,
                     col_name=['log_ts_pre', 'trace_ts_pre', 'trace_ts_start']):
    """
    Standardize continuous features in test set, using mean and standard deviation calculated from training set.

    Parameters
    ----------
    df: pandas.Dataframe
        Event log
    mean_dict:dictionary
        The keys are feature names and the values are mean values.
    std_dict: dictionary
        The keys are feature names and the values are standard deviation values.
    column_name: list, optional
        Name(s) of column(s) containing continuous features.

    Returns
    -------
    df: pandas.Dataframe
        Dataframe with standardized continuous features.
    """
    # loop through all features
    for col in col_name:
        mean = mean_dict[col]
        std = std_dict[col]
        
        # standardize the column
        df[col] = (df[col] - mean) / std
    
    return df

def mapping_case_id(df, 
                    case_id = 'case:concept:name'):
    """
    
    Create a dictionary that stores the one-to-one mapping of case_id and index, then transform the original case ID to index.  
    
    Parameters
    ----------
    df: pandas.DataFrame
        Event log 
    case_id: str, optional
        Name of column containing case ID

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with case ID transformed into index.
    case_id_dict: dictionary
        Store the one-to-one mapping of case_id and index
    
    """
    # create the mapping dictionary
    case_id_dict = {}
    n = 1
    for id in df[case_id].unique():
        case_id_dict[id] = n
        n += 1
    
    # map case ID to index
    df[case_id] = df[case_id].map(case_id_dict)

    return df, case_id_dict

def add_soc_eoc(df,
                case_id = 'case:concept:name', timestamp = 'time:timestamp', event_name = 'concept:name'):
    
    """
    
    Create rows containing SOC (Start of Case) token and EOC (End of Case) token. 
    
    Parameters
    ----------
    df: pandas.DataFrame
        Event log 
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp
    event_name: str, optional
        Name of column containing activity label

    Returns
    -------
    df: pandas.DataFrame
        Two rows are added for each case, and one column is added
    
    """
    # sort dataframe by case_id and timestamp
    df = df.sort_values(by=[case_id, timestamp]).reset_index(drop=True)

    # add SOC and EOC rows
    new_rows = []
    for i in range(len(df)):

        # if this is the first event in a case, append a soc_row before appending all events of the case
        if i == 0 or df[case_id].iloc[i] != df[case_id].iloc[i - 1]:

            # Create a new row where event name is 'SOC', and values in other columns are the same as the first event
            soc_row = df.iloc[i].copy() # This returns a series
            soc_row[event_name] = 'SOC'
            
            # Append the 'SOC' row 
            new_rows.append(soc_row)

        # append other rows in this case
        new_rows.append(df.iloc[i])

        # if this is the last event in a case, append a eoc_row after all events of the case
        if i == len(df) - 1 or df[case_id].iloc[i] != df[case_id].iloc[i + 1]:
            # create a new row where event name is 'EOC', and values in other columns are the same as the last event
            eoc_row = df.iloc[i].copy() # This returns a series
            eoc_row[event_name] = 'EOC'

            # Append the 'EOC' row 
            new_rows.append(eoc_row)

    new_df = pd.DataFrame(new_rows)

    # Create a helper column 'Order' to ensure that SOC is always immediately above the first event in a case, EOC is always immediately below the last event in a case
    new_df['Order'] = new_df.apply(lambda row: 
                                   row[case_id] * 10 + (1 if row[event_name] == 'SOC' 
                                                    else (3 if row[event_name] == 'EOC' 
                                                    else 2)), 
                                    axis=1)
    new_df = new_df.sort_values(by=[timestamp, 'Order']).reset_index(drop=True)
    # Drop the helper column
    new_df = new_df.drop(columns=['Order'])
    # Sort the dataframe
    new_df = new_df.sort_values(by=timestamp).reset_index(drop=True)

    # Create event index based on the chronological order 
    new_df['event_idx'] = range(1, len(new_df) + 1)

    return new_df

def train_mapping_event_name(df, event_name = 'concept:name'):
    """
    
    Mapping activity labels that appear in training set to index.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Training set
    event_name: str, optional
        Name of column containing activity label

    Returns
    -------
    df: pandas.DataFrame
        Event log
    event_name_dict: dictionary
        The keys are activity labels in training set and the values are the corresponding index.
    
    """
    # initialize dictionary
    event_name_dict = {"SOC":2, 
                       "EOC":3}
    
    # create the mapping dictionary
    n = int(4)
    for name in df[event_name].unique():
        if name not in event_name_dict.keys():
            event_name_dict[name] = n
            n += 1
    
    # map activity label to index
    df[event_name] = df[event_name].map(event_name_dict)

    return df, event_name_dict


def test_mapping_event_name(df, event_name_dict,
                            event_name = 'concept:name'):
    """
    
    Mapping activity labels that appear in test set to index.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Event log 
    event_name_dict: dictionary
        The keys are activity labels in training set and the values are the corresponding index.
    event_name: str, optional
        Name of column containing activity label

    Returns
    -------
    df: pandas.DataFrame
        Event log
    event_name_dict: dictionary
        The keys are activity labels in training set and the values are the corresponding index.
    
    """
    # initialize the dictionary
    test_event_name_dict = event_name_dict.copy()
    
    # create the mapping dictionary
    for name in df[event_name].unique():
        # activity labels not appearing in training set will be assigned index 1
        if name not in test_event_name_dict.keys():
            test_event_name_dict[name] = int(1)
    
    # map activity label to index
    df[event_name] = df[event_name].map(test_event_name_dict)

    return df, test_event_name_dict

