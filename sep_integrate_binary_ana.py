"""
file name: sep_integrate_binary_ana.py
Author: Alex

This code works on analyzing the relationship between distance and classification performance on binary model.
"""
import csv
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_recall_fscore_support
from sklearn import tree
from kl_cal import kl_calculator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ttest_ind
from scipy.spatial import distance


def sep_inte(model_type, n_reps):
    """
    This function carries out the analysis of binary model
    :param model_type: Type of model being used
    :param n_reps: number of replication
    """
    # csv file and log file set up
    log_file = 'log/' + model_type + '_' + 'bin' + '_' + 'ana' + '.txt'
    csv_file = 'results/' + model_type + '_' + 'bin' + '_' + 'ana' + '.csv'
    sys.stdout = open(log_file, "w")

    # initialize metric dictionary
    record_metrics = ['y', 'y_hat', 'mah_p', 'mah_f']
    metrics = {key: [] for key in record_metrics}

    # read data
    df = pd.read_csv("predictive_maintenance.csv")
    df['type'] = df['type'].astype('category').cat.codes
    df['failure.type'] = df['failure.type'].astype('category')
    df['failure.type'].cat.set_categories(['No Failure', 'Heat Dissipation Failure', 'Power Failure',
                                           'Overstrain Failure', 'Tool Wear Failure',
                                           'Random Failures'], inplace=True)
    df['failure.type'] = df['failure.type'].astype('category').cat.codes

    y = df['target']
    X = df.drop(columns='target')

    # replicate
    for i in range(n_reps):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=X['failure.type'],
                                                            random_state=i)
        model = None
        if model_type == 'lr':
            model = LogisticRegression(max_iter=1e4, solver='saga')
        elif model_type == 'svm':
            model = SVC()
        elif model_type == 'gnb':
            model = GaussianNB()
        elif model_type == 'lda':
            model = LinearDiscriminantAnalysis()
        else:
            print("Error: model does not exist")

        # training
        model.fit(X_train.drop(columns='failure.type'), y_train)

        # testing
        # y_train_pred = model.predict(X_train.drop(columns='failure.type'))
        y_test_pred = model.predict(X_test.drop(columns='failure.type'))

        # calculate distance
        dist_p = X_train[(X_train['failure.type'] == 0)].drop(columns='failure.type')
        dist_f = X_train[(X_train['failure.type'] != 0)].drop(columns='failure.type')
        for point in X_test.drop(columns='failure.type').to_numpy():
            # mahalanobis
            mean_p = np.mean(dist_p, axis=0)
            cov_p = np.cov(dist_p.T)  # Transpose data before calculating covariance
            cov_p_inv = np.linalg.inv(cov_p)  # Calculate the inverse of the covariance matrix
            mah_p = distance.mahalanobis(point, mean_p, cov_p_inv)

            mean_f = np.mean(dist_f, axis=0)
            cov_f = np.cov(dist_f.T)  # Transpose data before calculating covariance
            cov_f_inv = np.linalg.inv(cov_f)  # Calculate the inverse of the covariance matrix
            mah_f = distance.mahalanobis(point, mean_f, cov_f_inv)

            # record
            metrics['mah_p'].append(mah_p)
            metrics['mah_f'].append(mah_f)
        metrics['y'] = y_test.values
        metrics['y_hat'] = y_test_pred

    # output dictionary to a csv file
    results_df = pd.DataFrame()
    for key, value in metrics.items():
        results_df[key] = value

    results_df.to_csv(csv_file, index=False)
    sys.stdout.close()


def main():
    """
    Main function
    """
    n_reps = 1
    models = ['lr']

    for model in models:
        sep_inte(model, n_reps)


if __name__ == "__main__":
    main()
