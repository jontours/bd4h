import models_partc
import models_partb
#from sklearn.cross_validation import KFold, ShuffleSplit
from sklearn.model_selection import KFold, ShuffleSplit
import numpy as np
import pandas as pd
from numpy import mean
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	kf = KFold(n_splits=k)
	log_regressor = LogisticRegression()
	scores = []
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		log_regressor.fit(X_train, y_train)
		y_predict = log_regressor.predict(X_test)
		acc, auc_, precision, recall, f1score = models_partc.classification_metrics(y_predict, y_test)
		scores.append([acc, auc_])
	scores_frame = pd.DataFrame(scores)
	return scores_frame[0].mean(), scores_frame[1].mean()


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations

	sf = ShuffleSplit(n_splits=5, test_size=test_percent)
	log_regressor = LogisticRegression()
	scores = []
	for train_index, test_index in sf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		log_regressor.fit(X_train, y_train)
		y_predict = log_regressor.predict(X_test)
		acc, auc_, precision, recall, f1score = models_partc.classification_metrics(y_predict, y_test)
		scores.append([acc, auc_])
	scores_frame = pd.DataFrame(scores)
	return scores_frame[0].mean(), scores_frame[1].mean()


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

