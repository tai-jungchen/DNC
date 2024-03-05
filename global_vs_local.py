"""
file name: global_vs_local.py
Author: Alex

Test the performance of Global model, which is trained on all data, and the local models,
which are trained on majority-minority_sub data. Note that the testing data for both global and local models are
majority-minority_sub data to make a fair comparison.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

REPS = 30    # number of train-test split replication


def main():
    """
    Carry out the comparison.
    """
    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")

    subs = [1, 2, 3, 4, 5]

    for sub in subs:
        record_metrics = ['acc', 'bacc', 'prec', 'rec', 'spec', 'f1', 'npv']
        metrics_df_global = {key: [] for key in record_metrics}
        metrics_df_local = {key: [] for key in record_metrics}

        for i in range(REPS):
            result_helper(df, 'RF', i, sub, metrics_df_global, metrics_df_local)


def result_helper(data, model_type, rand, sub, metrics_global, metrics_local):
    """
    Conduct AdaBoost on the data given the classification type
    :param metrics_local: local performance storage
    :type metrics_local: dict
    :param metrics_global:global performance storage
    :type metrics_global: dict
    :param rand: random state
    :type rand: int
    :param sub: subclass
    :type sub: int
    :param model_type: type of model used to classify
    :type: model_type: String
    :param data: original data
    :type data: pandas dataframe
    """
    # logging setup #
    if rand == 0:
        log_file = 'log/global_vs_local/' + model_type + "_" + str(sub) + '.txt'
        sys.stdout = open(log_file, "w")

    # initialize the classifier #
    global_clf = None
    local_clf = None
    if model_type == 'XGBOOST':
        global_clf = xgb.XGBClassifier()
        local_clf = xgb.XGBClassifier()
    elif model_type == 'adaBoost':
        base_classifier = DecisionTreeClassifier()
        global_clf = AdaBoostClassifier(estimator=base_classifier, random_state=42, n_estimators=50)
        local_clf = AdaBoostClassifier(estimator=base_classifier, random_state=42, n_estimators=50)
    elif model_type == 'LR':
        global_clf = LogisticRegression(max_iter=1000)
        local_clf = LogisticRegression(max_iter=1000)
    elif model_type == 'DT':
        global_clf = DecisionTreeClassifier()
        local_clf = DecisionTreeClassifier()
    elif model_type == 'RF':
        global_clf = RandomForestClassifier()
        local_clf = RandomForestClassifier()

    # global train #
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-4], data['target'], test_size=0.3,
                                                        stratify=data['failure.type'], random_state=rand)
    global_clf.fit(X_train, y_train)

    # local train #
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-4], data['failure.type'], test_size=0.3,
                                                        stratify=data['failure.type'], random_state=rand)
    # select only majority and minority sub
    X_train = X_train[(y_train == sub) | (y_train == 0)]
    y_train = y_train[(y_train == sub) | (y_train == 0)]
    y_train[y_train != 0] = 1  # turn non-zero sub minority into 1

    local_clf.fit(X_train, y_train)

    # test for both models
    X_test = X_test[(y_test == sub) | (y_test == 0)]
    y_test = y_test[(y_test == sub) | (y_test == 0)]
    y_test[y_test != 0] = 1

    y_pred_train_global = global_clf.predict(X_train)
    y_pred_global = global_clf.predict(X_test)
    y_pred_train_local = local_clf.predict(X_train)
    y_pred_local = local_clf.predict(X_test)

    # performance
    print("Global model", sub)
    # print("Training:")
    # print(confusion_matrix(y_train, y_pred_train_global))
    # print(classification_report(y_train, y_pred_train_global))
    # print("Testing:")
    print(confusion_matrix(y_test, y_pred_global))
    print(classification_report(y_test, y_pred_global))

    print("Local model", sub)
    # print("Training:")
    # print(confusion_matrix(y_train, y_pred_train_local))
    # print(classification_report(y_train, y_pred_train_local))
    # print("Testing:")
    print(confusion_matrix(y_test, y_pred_local))
    print(classification_report(y_test, y_pred_local))

    # Global performance #
    acc = accuracy_score(y_test, y_pred_global)
    bacc = balanced_accuracy_score(y_test, y_pred_global)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_global)
    tn, fp, fn, tp = np.ravel(confusion_matrix(y_test, y_pred_global))
    spec = tn / (tn + fp)
    npv = tn / (tn + fn)
    # Store performance
    metrics_global['acc'].append(round(acc, 4))
    metrics_global['bacc'].append(round(bacc, 4))
    metrics_global['prec'].append(round(precision[1], 4))
    metrics_global['rec'].append(round(recall[1], 4))
    metrics_global['spec'].append(round(spec, 4))
    metrics_global['npv'].append(round(npv, 4))
    metrics_global['f1'].append(round(f1[1], 4))

    # local performance #
    acc = accuracy_score(y_test, y_pred_local)
    bacc = balanced_accuracy_score(y_test, y_pred_local)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_local)
    tn, fp, fn, tp = np.ravel(confusion_matrix(y_test, y_pred_local))
    spec = tn / (tn + fp)
    npv = tn / (tn + fn)
    # Store performance
    metrics_local['acc'].append(round(acc, 4))
    metrics_local['bacc'].append(round(bacc, 4))
    metrics_local['prec'].append(round(precision[1], 4))
    metrics_local['rec'].append(round(recall[1], 4))
    metrics_local['spec'].append(round(spec, 4))
    metrics_local['npv'].append(round(npv, 4))
    metrics_local['f1'].append(round(f1[1], 4))

    if rand == (REPS - 1):
        # save output to
        filename_global = "results/global_vs_local/" + "global_" + model_type + str(sub) + ".csv"
        filename_local = "results/global_vs_local/" + "local_" + model_type + str(sub) + ".csv"
        save_metrics(metrics_global, filename_global, "global")
        save_metrics(metrics_local, filename_local, "local")


def save_metrics(dic, filename, global_or_local):
    """
    Save the metric dictionary into .csv file
    :param dic: the metric dictionary
    :type dic: dict
    :param filename: the filename to be used for the .csv file
    :type filename: String
    :param global_or_local: Whether the model is global or local
    :type global_or_local: String
    """
    results_df = pd.DataFrame()
    print("\nAverage Performance for", global_or_local, "model:")
    for key, value in dic.items():
        print(f'{key}: {round(float(np.mean(value)), 4)}, S.E.: {round(np.std(value) / len(value), 4)}')
        results_df[key] = value

    for key, value in dic.items():
        results_df.at[len(value), key] = round(float(np.mean(value)), 4)
        results_df.at[len(value) + 1, key] = round(np.std(value) / len(value), 4)
    results_df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
