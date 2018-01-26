import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.decomposition import TruncatedSVD


import utils

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT
# USE THIS RANDOM STATE FOR ALL OF THE PREDICTIVE MODELS
# THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: X_train, Y_train and X_test
#output: Y_pred
def logistic_regression_pred(X_train, Y_train, X_test):
	#TODO: train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	lrp = LogisticRegression(random_state=RANDOM_STATE)
	lrp.fit(X_train, Y_train)
	y_predict = lrp.predict(X_test)
	return y_predict

#input: X_train, Y_train and X_test
#output: Y_pred
def svm_pred(X_train, Y_train, X_test):
	#TODO:train a SVM classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier
	svm_predictor = LinearSVC(random_state=RANDOM_STATE)
	svm_predictor.fit(X_train, Y_train)
	y_predict = svm_predictor.predict(X_test)

	return y_predict

#input: X_train, Y_train and X_test
#output: Y_pred
def decisionTree_pred(X_train, Y_train, X_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.
	clf_gini = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5)
	clf_gini.fit(X_train, Y_train)
	y_predict = clf_gini.predict(X_test)

	return y_predict


def linearRegression_pred(X_train, Y_train, X_test):
	svd = TruncatedSVD(n_components=200)
	train_features = svd.fit_transform(X_train)
	rfr = LogisticRegression(random_state=RANDOM_STATE, max_iter=50)
	#rfr = RandomForestClassifier(n_estimators=100, n_jobs=1,
	#							 random_state=2016, verbose=1,
	#							 class_weight='balanced', oob_score=True)

	rfr.fit(train_features, Y_train)

	test_features = svd.transform(X_test)
	y_predict = rfr.predict(test_features)
	return y_predict



#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	# NOTE: It is important to provide the output in the same order
	accuracy = accuracy_score(Y_true, Y_pred)
	auc_ret = roc_auc_score(Y_true, Y_pred)
	prec = precision_score(Y_true, Y_pred)
	recall = recall_score(Y_true, Y_pred)
	f1_score_ret = f1_score(Y_pred, Y_true)
	return accuracy, auc_ret, prec, recall, f1_score_ret

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
	print "______________________________________________"
	print "Classifier: "+classifierName
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print "Accuracy: "+str(acc)
	print "AUC: "+str(auc_)
	print "Precision: "+str(precision)
	print "Recall: "+str(recall)
	print "F1-score: "+str(f1score)
	print "______________________________________________"
	print ""

def main():
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("../data/features_svmlight.validate")

	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("SVM",svm_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train,X_test),Y_test)
	#display_metrics("Random Forrest", linearRegression_pred(X_train, Y_train, X_test), Y_test)

if __name__ == "__main__":
	main()
	
