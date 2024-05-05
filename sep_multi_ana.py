"""
file name: sep_multi_ana.py
Author: Alex

This code tries to verify whether better separation implies better classification performance by comparing whether
training on all instance yields better classification performance or training on subclass of minority class and pass
samples will have better classification performance. The feature used here is multi-variate.
"""
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


def sep_multi(model_type, sub, n_reps):
    """
    This function carries out the training and testing of two types of models, one trained on all the instance and the
    other trained on only the particular failure type and pass instance
    :param model_type: Type of model being used
    :param sub: the failure type
    :param dist: the distance metric used
    :param n_reps: number of replication
    """
    # csv file and log file set up
    log_file = 'log/' + model_type + '_' + str(sub) + '_' + 'multi' + '.txt'
    csv_file = 'results/' + model_type + '_' + str(sub) + '_' + 'multi' + '.csv'
    sys.stdout = open(log_file, "w")

    # initialize metric dictionary
    record_metrics = ['mah_bin', 'prec_bin', 'rec_bin', 'spec_bin', 'bacc_bin', 'acc_bin', 'f1_bin', 'kappa_bin',
                      'mah_sub', 'prec_sub', 'rec_sub', 'spec_sub', 'bacc_sub', 'acc_sub', 'f1_sub', 'kappa_sub']
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
        bin_model = None
        model = None
        if model_type == 'lr':
            bin_model = LogisticRegression(max_iter=1e4, solver='saga')
            model = LogisticRegression(max_iter=1e4, solver='saga')
        elif model_type == 'svm':
            bin_model = SVC()
            model = SVC()
        elif model_type == 'gnb':
            bin_model = GaussianNB()
            model = GaussianNB()
        elif model_type == 'lda':
            bin_model = LinearDiscriminantAnalysis()
            model = LinearDiscriminantAnalysis()
        else:
            print("Error: model does not exist")

        # calculate distance
        # mahalanobis
        mah_bin = mahalanobis(X_train[(X_train['failure.type'] == 0)].drop(columns='failure.type'),
                                        X_train[(X_train['failure.type'] != 0)].drop(columns='failure.type'))
        metrics['mah_bin'].append(round(mah_bin, 4))
        mah_sub = mahalanobis(X_train[(X_train['failure.type'] == 0)].drop(columns='failure.type'),
                                        X_train[(X_train['failure.type'] == sub)].drop(columns='failure.type'))
        metrics['mah_sub'].append(round(mah_sub, 4))

        # partition the training and testing sets
        condition_train = (X_train['failure.type'] == sub) | (X_train['failure.type'] == 0)
        X_train_fk = X_train[condition_train].drop(columns='failure.type')
        y_train_fk = y_train[condition_train]

        condition_test = (X_test['failure.type'] == sub) | (X_test['failure.type'] == 0)
        X_test_fk = X_test[condition_test].drop(columns='failure.type')
        y_test_fk = y_test[condition_test]

        # training
        bin_model.fit(X_train.drop(columns='failure.type'), y_train)
        model.fit(X_train_fk, y_train_fk)

        # testing
        # y_train_pred_bin = bin_model.predict(X_train.drop(columns='failure.type'))
        y_test_pred_bin = bin_model.predict(X_test_fk)
        # y_train_pred_sub = model.predict(X_train_fk)
        y_test_pred_sub = model.predict(X_test_fk)

        # binary model classification performance
        acc_bin = accuracy_score(y_test_fk, y_test_pred_bin)
        kappa_bin = cohen_kappa_score(y_test_fk, y_test_pred_bin)
        bacc_bin = balanced_accuracy_score(y_test_fk, y_test_pred_bin)
        prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(y_test_fk, y_test_pred_bin)
        tn, fp, fn, tp = np.ravel(confusion_matrix(y_test_fk, y_test_pred_bin))
        spec_bin = tn / (tn + fp)

        # sub model classification performance
        acc_sub = accuracy_score(y_test_fk, y_test_pred_sub)
        kappa_sub = cohen_kappa_score(y_test_fk, y_test_pred_sub)
        bacc_sub = balanced_accuracy_score(y_test_fk, y_test_pred_sub)
        prec_sub, rec_sub, f1_sub, _ = precision_recall_fscore_support(y_test_fk, y_test_pred_sub)
        tn, fp, fn, tp = np.ravel(confusion_matrix(y_test_fk, y_test_pred_sub))
        spec_sub = tn / (tn + fp)

        # performance
        print("Binary model")
        print(confusion_matrix(y_test_fk, y_test_pred_bin, labels=[0, 1]))
        print(classification_report(y_test_fk, y_test_pred_bin))
        metrics['prec_bin'].append(round(prec_bin[1], 4))
        metrics['rec_bin'].append(round(rec_bin[1], 4))
        metrics['spec_bin'].append(round(spec_bin, 4))
        metrics['bacc_bin'].append(round(bacc_bin, 4))
        metrics['acc_bin'].append(round(acc_bin, 4))
        metrics['f1_bin'].append(round(f1_bin[1], 4))
        metrics['kappa_bin'].append(round(kappa_bin, 4))

        print("\nSub model")
        print(confusion_matrix(y_test_fk, y_test_pred_sub, labels=[0, 1]))
        print(classification_report(y_test_fk, y_test_pred_sub))
        metrics['prec_sub'].append(round(prec_sub[1], 4))
        metrics['rec_sub'].append(round(rec_sub[1], 4))
        metrics['spec_sub'].append(round(spec_sub, 4))
        metrics['bacc_sub'].append(round(bacc_sub, 4))
        metrics['acc_sub'].append(round(acc_sub, 4))
        metrics['f1_sub'].append(round(f1_sub[1], 4))
        metrics['kappa_sub'].append(round(kappa_sub, 4))

    # performance summary
    results_df = pd.DataFrame()
    for key, value in metrics.items():
        print(f'{key}: {np.mean(value)}, S.E.: {round(np.std(value) / len(value), 4)}')
        results_df.at[0, key] = round(float(np.mean(value)), 4)
        results_df.at[1, key] = round(np.std(value) / len(value), 4)
    results_df.to_csv(csv_file, index=False)
    sys.stdout.close()


def mahalanobis(data, points):
    """
    Calculate the mahalanobis distance for all points to the data
    :param data: the distribution
    :param points: the points
    :return: the average mahalanobis distance for all points
    """
    # Calculate mean and covariance matrix of the data
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)  # Transpose data before calculating covariance
    cov_inv = np.linalg.inv(cov)    # Calculate the inverse of the covariance matrix
    mahalanobis_dists = 0

    # Point for which Mahalanobis distance needs to be calculated
    for point in points.to_numpy():
        # Calculate Mahalanobis distance using scipy.spatial.distance.mahalanobis
        mahalanobis_dist = distance.mahalanobis(point, mean, cov_inv)
        mahalanobis_dists += mahalanobis_dist
    return mahalanobis_dists / len(points)


def main():
    """
    Main function
    """
    n_reps = 30
    subs = [1, 2, 3, 4, 5]
    models = ['gnb']

    for model in models:
        for sub in subs:
            sep_multi(model, sub, n_reps)


if __name__ == "__main__":
    main()
