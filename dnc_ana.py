"""
file name: dnc_ana.py
Author: Alex

This code analyze the DNC method. The analysis mainly focuses on finding out which local models gave mis-classification.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_recall_fscore_support


def main(model_type, dataset):
    # read data
    if dataset == 'MPMC':
        df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
        X_train, X_test, y_train, y_test_multi = train_test_split(df.iloc[:, :-4], df['failure.type'], test_size=0.3,
                                                            stratify=df['failure.type'], random_state=0)
    elif dataset == 'MNIST':
        df = pd.read_csv("datasets/preprocessed/mnist_imb.csv")
        X_train, X_test, y_train, y_test_multi = train_test_split(df.iloc[:, :-2], df['label'], test_size=0.3,
                                                            stratify=df['label'], random_state=0)
    else:
        raise Exception("Invalid dataset.")

    # training
    models = []
    subs = [i for i in range(1, len(y_train.value_counts()))]
    for sub in subs:
        if model_type == 'LR':
            model = LogisticRegression(max_iter=100)
        elif model_type == 'DT':
            model = DecisionTreeClassifier()
        elif model_type == 'RF':
            model = RandomForestClassifier()
        elif model_type == 'XGB':
            model = xgb.XGBClassifier()
        else:
            raise Exception("Invalid model type.")

        condition = (y_train == sub) | (y_train == 0)
        X_train_sub = X_train[condition]
        y_train_sub = y_train[condition]
        model.fit(X_train_sub, y_train_sub)
        models.append(model)

    # testing
    y_multi_preds = []
    for md in models:
        y_pred = md.predict(X_test)
        y_multi_preds.append(y_pred)

    # voting
    y_multi_preds = np.array(y_multi_preds)
    y_pred = np.where(np.sum(y_multi_preds, axis=0) > 0, 1, 0)

    y_test = y_test_multi.copy()
    y_test[y_test != 0] = 1

    print(f'DNC {model_type}')
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main("LR", "MPMC")
