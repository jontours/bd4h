import utils
import etl
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

# Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

RANDOM_STATE = 545510477

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''


def my_features():
    # TODO: complete this
    train_path = '../data/test/'
    deliverables_path = '../deliverables/'
    # Calculate index date
    events = pd.read_csv(train_path + 'events.csv')
    feature_map = pd.read_csv(train_path + 'event_feature_map.csv')
    # Aggregate the event values for each pat ient
    aggregated_events = etl.aggregate_events(events, None, feature_map, deliverables_path)

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

    line = ''
    line_svm = ''
    for key, value in sorted(patient_features.iteritems()):
        line += str(int(key)) + ' '
        line_svm += str(1) + ' '
        value = sorted(value)
        for item in value:
            line += str(int(item[0])) + ":" + str(format(item[1], '.6f')) + ' '
            line_svm += str(int(item[0])) + ":" + str(format(item[1], '.6f')) + ' '
        line += '\n'
        line_svm += '\n'

    deliverable2 = open(deliverables_path + 'test_features.txt', 'wb')
    deliverable2.write(line)
    deliverable2.close()

    svm_file = open(deliverables_path + 'test_mymodel_features.train', 'wb')
    svm_file.write(line_svm)
    svm_file.close()

    data_train = load_svmlight_file(deliverables_path + 'test_mymodel_features.train', n_features=3190)
    X_test = data_train[0]
    print(X_test.shape)

    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")

    return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''


def my_classifier_predictions(X_train, Y_train, X_test):
    svd = TruncatedSVD(n_components=200)
    train_features = svd.fit_transform(X_train)
    rfr = LogisticRegression(random_state=RANDOM_STATE, max_iter=50)
    # rfr = RandomForestClassifier(n_estimators=100, n_jobs=1,
    #							 random_state=2016, verbose=1,
    #							 class_weight='balanced', oob_score=True)

    rfr.fit(train_features, Y_train)

    test_features = svd.transform(X_test)
    y_predict = rfr.predict(test_features)
    return y_predict


def main():
    X_train, Y_train, X_test = my_features()
    Y_pred = my_classifier_predictions(X_train, Y_train, X_test)
    print Y_pred.shape
    utils.generate_submission("../deliverables/test_features.txt", Y_pred)


# The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()
