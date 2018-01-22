import utils
import pandas as pd

pd.options.mode.chained_assignment = None


# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    mortality['timestamp'] = pd.to_datetime(mortality['timestamp'])
    mortality = mortality.assign(indx_date=lambda x: x.timestamp - pd.DateOffset(days=30))
    mortality = mortality[['patient_id','indx_date']]
    dead_frame = pd.merge(events, mortality, left_on='patient_id', right_on='patient_id')
    alive_frame = events[~events['patient_id'].isin(dead_frame.patient_id.unique())]
    collated_dates = alive_frame.groupby(['patient_id'])['patient_id','timestamp'].max()
    collated_frame = pd.DataFrame(collated_dates)
    collated_frame.rename(columns = {'timestamp':'indx_date'}, inplace = True)
    indx_date = pd.concat([collated_frame, mortality])
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    events = pd.merge(events, indx_date, left_on='patient_id', right_on='patient_id')
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    events['indx_date'] = pd.to_datetime(events['indx_date'])
    filtered_events = events[(events['timestamp'] >= events['indx_date'] - pd.DateOffset(days=2000)) & (events['timestamp'] <= events['indx_date']) ]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'],
                           index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    # ---- for lab rows ----
    columns=['patient_id', 'feature_id', 'feature_value']
    filtered_events_df_lab = filtered_events_df[filtered_events_df['event_id'].str.contains('LAB')]
    filtered_events_df_lab['event_id'] = filtered_events_df_lab['event_id'].map(feature_map_df.set_index('event_id')['idx'])
    filtered_events_df_lab = filtered_events_df_lab.dropna(subset=['value'])
    aggregated_events_lab = filtered_events_df_lab.groupby(['patient_id', 'event_id'], as_index=False).count()
    aggregated_events_lab.rename(columns={'event_id':'feature_id'}, inplace=True)
    aggregated_events_lab_max = aggregated_events_lab.groupby(['feature_id'], as_index=False).agg({"value":"max"})
    merged_lab = pd.merge(aggregated_events_lab, aggregated_events_lab_max, left_on="feature_id", right_on="feature_id")
    merged_lab['feature_value'] = merged_lab['value_x'] / merged_lab['value_y']
    merged_lab = merged_lab[columns]


    # ---- regular continuous valued DRUG etc. stuff
    continuous_events = filtered_events_df[filtered_events_df['event_id'].str.contains('DRUG') | filtered_events_df['event_id'].str.contains('DIAG')]
    continuous_events['event_id'] = continuous_events['event_id'].map(
        feature_map_df.set_index('event_id')['idx'])
    continuous_events = continuous_events.dropna(subset=['value'])
    aggregated_events = continuous_events.groupby(['patient_id','event_id'], as_index=False).agg({"value":"sum"})
    aggregated_events.rename(columns={'event_id': 'feature_id', 'value':'feature_value'}, inplace=True)
    aggregated_events_max = aggregated_events.groupby(['feature_id'], as_index=False).agg({"feature_value":"max"})
    merged = pd.merge(aggregated_events, aggregated_events_max, left_on="feature_id", right_on="feature_id")
    merged['feature_value'] = merged['feature_value_x'] / merged['feature_value_y']
    merged = merged[columns]
    aggregated_events = pd.concat([merged_lab, merged])
    #aggregated_events = aggregated_events.dropna()
    #print(aggregated_events)
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv',
                             columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    patient_features = {}
    for index, row in aggregated_events.iterrows():
        if not patient_features.get(row['patient_id']):

            patient_features[row['patient_id']] = [(row['feature_id'], row['feature_value'])]
        else:
            patient_features[row['patient_id']].append((row['feature_id'], row['feature_value']))
    mortality_dict = {}
    for index, row in mortality.iterrows():
        mortality_dict[row['patient_id']] = row['label']

    for key in patient_features.keys():
        if not key in mortality_dict:
            mortality_dict[key] = 0


    return patient_features, mortality_dict

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    line = ''
    for key, value in sorted(patient_features.iteritems()):
        line += str(int(key)) + ' ' + str(mortality[key]) + ' '
        value = sorted(value)
        for item in value:
            line += str(int(item[0])) + ":" + str(format(item[1], '.6f')) + ' '
        line += '\n'
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    deliverable1.write(line)
    deliverable2.write(line)

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()