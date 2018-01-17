import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    dead_frame = pd.merge(events, mortality, left_on='patient_id', right_on='patient_id')
    alive_frame = events[~events['patient_id'].isin(dead_frame.patient_id.unique())]
    avg_dead_event_count = dead_frame.groupby(['patient_id'])['patient_id'].agg(['count']).mean()['count']
    max_dead_event_count = dead_frame.groupby(['patient_id'])['patient_id'].agg(['count']).max()['count']
    min_dead_event_count = dead_frame.groupby(['patient_id'])['patient_id'].agg(['count']).min()['count']
    avg_alive_event_count = alive_frame.groupby(['patient_id'])['patient_id'].agg(['count']).mean()['count']
    max_alive_event_count = alive_frame.groupby(['patient_id'])['patient_id'].agg(['count']).max()['count']
    min_alive_event_count = alive_frame.groupby(['patient_id'])['patient_id'].agg(['count']).min()['count']

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    dead_frame = pd.merge(events, mortality, left_on='patient_id', right_on='patient_id')
    alive_frame = events[~events['patient_id'].isin(dead_frame.patient_id.unique())]

    avg_dead_encounter_count = dead_frame.groupby(['patient_id']).timestamp_x.nunique().mean()
    max_dead_encounter_count = dead_frame.groupby(['patient_id']).timestamp_x.nunique().max()
    min_dead_encounter_count = dead_frame.groupby(['patient_id']).timestamp_x.nunique().min()
    avg_alive_encounter_count = alive_frame.groupby(['patient_id']).timestamp.nunique().mean()
    max_alive_encounter_count = alive_frame.groupby(['patient_id']).timestamp.nunique().max()
    min_alive_encounter_count = alive_frame.groupby(['patient_id']).timestamp.nunique().min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    mortality['timestamp'] = pd.to_datetime(mortality['timestamp'])
    dead_frame = pd.merge(events, mortality, left_on='patient_id', right_on='patient_id')
    alive_frame = events[~events['patient_id'].isin(dead_frame.patient_id.unique())]
    collated_dates = dead_frame.groupby(['patient_id']).apply(lambda x: x.timestamp_x.max() - x.timestamp_x.min()).astype('timedelta64[D]').astype(int)
    alive_collated_dates = alive_frame.groupby(['patient_id']).apply(lambda x: x.timestamp.max() - x.timestamp.min()).astype('timedelta64[D]').astype(int)
    avg_dead_rec_len = collated_dates.mean()
    max_dead_rec_len = collated_dates.max()
    min_dead_rec_len = collated_dates.min()
    avg_alive_rec_len = alive_collated_dates.mean()
    max_alive_rec_len = alive_collated_dates.max()
    min_alive_rec_len = alive_collated_dates.min()
    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
