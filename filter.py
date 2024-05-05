"""
Author: Alex

This code implements the filtering model (not using at the moment)
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn import tree
from metric_helper import metrics

MODEL_TYPE_1 = 'lr'
MODEL_TYPE_2 = 'rf'
SUB = 1
DECISION_THRESH = 0.98


def main():
	df = pd.read_csv("predictive_maintenance.csv")
	df['type'] = df['type'].astype('category').cat.codes
	df['failure.type'] = df['failure.type'].astype('category')
	df['failure.type'].cat.set_categories(['No Failure', 'Heat Dissipation Failure', 'Power Failure',
															  'Overstrain Failure', 'Tool Wear Failure',
															  'Random Failures' ], inplace=True)
	df['failure.type'] = df['failure.type'].astype('category').cat.codes

	X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['failure.type'], test_size=0.3,
														random_state=11, stratify=df['failure.type'])
	model = None

	X = df.drop(columns=['failure.type'])
	y = df['failure.type']
	betas = []

	x1 = X.iloc[:, 0].to_numpy()
	x2 = X.iloc[:, 1].to_numpy()
	x = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)

	for X_town in X.columns:
		x = X[X_town].to_numpy()
		model = LinearRegression().fit(x.reshape(-1, 1), y)
		beta = model.coef_
		betas.append(beta)

	# Phase I
	if MODEL_TYPE_1 is 'lr':
		model = LogisticRegression(max_iter=1e4, multi_class='multinomial')
		# clf = LogisticRegression(max_iter=1e4)
		# model = OneVsOneClassifier(clf)
	elif MODEL_TYPE_1 is 'svm':
		model = SVC(kernel='linear', C=0.5, decision_function_shape='ovo')
	elif MODEL_TYPE_1 is 'gnb':
		model = GaussianNB()
	elif MODEL_TYPE_1 is 'dt':
		model = DecisionTreeClassifier(max_depth=3)
	elif MODEL_TYPE_1 is 'rf':
		model = RandomForestClassifier(random_state=11)
	elif MODEL_TYPE_1 is 'knn':
		model = KNeighborsClassifier(n_neighbors=6)
		# scores = cross_val_score(knn, X, y, cv=5)
	else:
		print("Error: model does not exist")
	model.fit(X_train, y_train)
	# tree.plot_tree(model)
	# plt.show()

	X_test_1 = X_test[(y_test == 0) | (y_test == SUB)]
	y_test_1 = y_test[(y_test == 0) | (y_test == SUB)]

	y_pred_1 = tuned_prediction(model, X_test_1, DECISION_THRESH, y_test_1)
	# y_pred_1 = model.predict(X_test_1)
	print(MODEL_TYPE_1)
	# print(confusion_matrix(y_test, y_pred, labels=[1, 0]))
	print(confusion_matrix(y_test_1, y_pred_1, labels=[0, 1, 2, 3, 4, 5]))
	# print(classification_report(y_test_1, y_pred_1))
	# print("kappa: ", round(cohen_kappa_score(y_test_1, y_pred_1), 4))

	# Phase II
	if MODEL_TYPE_2 is 'lr':
		model = LogisticRegression(max_iter=1e4)
		# clf = LogisticRegression(max_iter=1e4)
		# model = OneVsOneClassifier(clf)
	elif MODEL_TYPE_2 is 'svm':
		model = SVC(kernel='linear', C=0.5)
	elif MODEL_TYPE_2 is 'gnb':
		model = GaussianNB()
	elif MODEL_TYPE_2 is 'dt':
		model = DecisionTreeClassifier(max_depth=3)
	elif MODEL_TYPE_2 is 'rf':
		model = RandomForestClassifier(random_state=11)
	elif MODEL_TYPE_2 is 'knn':
		model = KNeighborsClassifier(n_neighbors=6)
		# scores = cross_val_score(knn, X, y, cv=5)
	else:
		print("Error: model does not exist")
	model.fit(X_train[(y_train == 0) | (y_train == SUB)], y_train[(y_train == 0) | (y_train == SUB)])

	y_pred_2 = model.predict(X_test_1[(y_pred_1 != 0) & (y_pred_1 != SUB)])
	y_test_2 = y_test_1[(y_pred_1 != 0) & (y_pred_1 != SUB)]
	print(MODEL_TYPE_2)
	# print(confusion_matrix(y_test, y_pred, labels=[1, 0]))
	print(confusion_matrix(y_test_2, y_pred_2, labels=[0, 1, 2, 3, 4, 5]))
	# print(classification_report(y_test_2, y_pred_2))
	# print("kappa: ", round(cohen_kappa_score(y_test_2, y_pred_2), 4))

	# Print the confusion matrix
	cm1 = confusion_matrix(y_test_1, y_pred_1, labels=[0, 1, 2, 3, 4, 5])

	# True negatives, false positives, false negatives and true positives
	# For the case of multi-class classification, these metrics should be used on a per-class basis
	tn1 = cm1[0, 0]
	tp1 = cm1[SUB, SUB]
	fn1 = cm1[SUB, 0]
	fp1 = cm1[0, SUB]

	cm2 = confusion_matrix(y_test_2, y_pred_2, labels=[0, 1, 2, 3, 4, 5])

	# True negatives, false positives, false negatives and true positives
	# For the case of multi-class classification, these metrics should be used on a per-class basis
	tn2 = cm2[0, 0]
	tp2 = cm2[SUB, SUB]
	fn2 = cm2[SUB, 0]
	fp2 = cm2[0, SUB]

	tn = tn1 + tn2
	fp = fp1 + fp2
	fn = fn1 + fn2
	tp = tp1 + tp2

	# bin_model = LogisticRegression(max_iter=1e4)
	bin_model = RandomForestClassifier(random_state=11)
	# bin_model = SVC(kernel='linear', C=0.5)
	bin_model.fit(X_train[(y_train == 0) | (y_train == SUB)], y_train[(y_train == 0) | (y_train == SUB)])
	y_pred = bin_model.predict(X_test_1)

	y_prob_bin = bin_model.predict_proba(X_test_1)

	print()
	print([tn, fp])
	print([fn, tp])

	print("\nFiltering model performance")
	metrics(tn, fp, fn, tp)

	# print(MODEL_TYPE)
	print("\nBinary model")
	print(confusion_matrix(y_test_1, y_pred))
	print(classification_report(y_test_1, y_pred))
	print("kappa: ", round(cohen_kappa_score(y_test_1, y_pred), 4))


def tuned_prediction(model, X, decision_threshold, y):
	"""
	Return the predction based on the given decision threshold
	:param model: <obj> The given model
	:param X: <np array> Features
	:param decision_threshold: <float> The given decision threshold
	:return: <np array> The predicted labels based on the given decision threshold
	"""
	y_prob = model.predict_proba(X)
	class_indices = np.argmax(model.predict_proba(X), axis=1)
	y_pred = np.where(np.max(y_prob, axis=1) > decision_threshold, np.argmax(model.predict_proba(X), axis=1), SUB+1)
	y_np = y.to_numpy()
	return y_pred.flatten()


if __name__ == "__main__":
	main()
