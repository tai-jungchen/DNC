"""
file name: binary_clf.py
Author: Alex

Test the performance of baseline binary classifier.
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, \
    precision_recall_fscore_support


def binary(model_type, n_reps, dataset):
    """
    This function carries out the binary classification.

    :param dataset: the dataset given
    :type dataset: String
    :param model_type: model type (LR, DT, RF, XGBOOST)
    :type model_type: String
    :param n_reps: number of train-test split replication
    :type n_reps: Integer
    """
    log_file = 'log/' + 'binary_' + dataset + '_' + model_type + '.txt'
    csv_file = 'results/' + 'binary_' + dataset + '_' + model_type + '.csv'
    sys.stdout = open(log_file, "w")

    record_metrics = ['acc', 'bacc', 'f1', 'precision', 'recall', 'specificity']
    metrics = {key: [] for key in record_metrics}

    # read data
    if dataset == 'maintenance':
        df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    elif dataset == 'nij':
        df = pd.read_csv("datasets/preprocessed/nij_data.csv")
    elif dataset == 'mnist':
        df = pd.read_csv("datasets/preprocessed/mnist_imb.csv")
    else:
        df = None

    for i in range(n_reps):
        if dataset == 'maintenance':
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-4], df['target'], test_size=0.3,
                                                                stratify=df['failure.type'], random_state=i)
        elif dataset == 'nij':
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['Recidivism'], test_size=0.3,
                                                                stratify=df['Recidivism_Year'], random_state=i)
        elif dataset == 'mnist':
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['label_bin'], test_size=0.3,
                                                                stratify=df['label'], random_state=i)
        else:
            X_train, X_test, y_train, y_test = None, None, None, None

        model = None
        if model_type == 'XGBOOST':
            model = xgb.XGBClassifier()
        elif model_type == 'LR':
            model = LogisticRegression(max_iter=10000)
        elif model_type == 'DT':
            model = DecisionTreeClassifier()
        elif model_type == 'RF':
            model = RandomForestClassifier()
        else:
            print("Error: model does not exist")

        model.fit(X_train, y_train)

        # testing #
        y_pred = model.predict(X_test)
        print(f'Binary {model_type}')
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(y_test, y_pred))

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
        tn, fp, fn, tp = np.ravel(confusion_matrix(y_test, y_pred))
        spec = tn / (tn + fp)
        # Store performance
        metrics['acc'].append(round(acc, 4))
        metrics['bacc'].append(round(bacc, 4))
        metrics['precision'].append(round(precision[1], 4))
        metrics['recall'].append(round(recall[1], 4))
        metrics['specificity'].append(round(spec, 4))
        metrics['f1'].append(round(f1[1], 4))

    # save output #
    results_df = pd.DataFrame()
    print("\nAverage Performance:")
    for key, value in metrics.items():
        print(f'{key}: {round(float(np.mean(value)), 4)}, S.E.: {round(np.std(value) / len(value), 4)}')
        results_df[key] = value

    for key, value in metrics.items():
        results_df.at[len(value), key] = round(float(np.mean(value)), 4)
        results_df.at[len(value) + 1, key] = round(np.std(value) / len(value), 4)
    results_df.to_csv(csv_file, index=False)
    sys.stdout.close()


def bin_analysis():
    """
    This function helps for analyzing the behavior of the binary classification.
    """
    # read data
    df = pd.read_csv("predictive_maintenance.csv")
    df['type'] = df['type'].astype('category').cat.codes
    df['failure.type'] = df['failure.type'].astype('category')
    df['failure.type'].cat.set_categories(['No Failure', 'Heat Dissipation Failure', 'Power Failure',
                                           'Overstrain Failure', 'Tool Wear Failure',
                                           'Random Failures'], inplace=True)
    df['failure.type'] = df['failure.type'].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['target'], test_size=0.3,
                                                        stratify=df['failure.type'])

    # Normalize coefficients using Z-score scaling
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_train)  # Scale your feature matrix
    # normalized_coefficients = model.coef_ / scaler.scale_

    model = LogisticRegression(max_iter=1e4, solver='saga')
    model.fit(X_train, y_train)

    feature_names = df.columns[:-2]
    feature_importance = model.coef_[0]  # / scaler.scale_
    # plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xticks(range(len(feature_importance)), feature_names, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Importance in Logistic Regression')
    plt.show()
