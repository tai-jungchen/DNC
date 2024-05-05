"""
file name: global_vs_local.py
Author: Alex

Test the performance of Global model, which is trained on all data, and the local models,
which are trained on majority-minority_sub data. Note that the testing data for both global and local models are
majority-minority_sub data to make a fair comparison.
"""
import sys
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

DATASET = "MNIST"


def main():
    """
    Carry out the comparison.
    """
    dataset = DATASET
    replications = 10
    # models = ["LR"]
    models = ["LR", "DT", "RF", "XGB"]

    if dataset == "MPMC":
        df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
        X = df.iloc[:, :-4]
        y_bin = df['target']
        y_multi = df['failure.type']
    elif dataset == "MNIST":
        df = pd.read_csv("datasets/preprocessed/mnist_imb.csv")
        X = df.iloc[:, :-2]
        y_bin = df['label_bin']
        y_multi = df['label']
    else:
        raise Exception("Invalid dataset!")

    for model in models:
        global_vs_local(X, y_bin, y_multi, model, reps=replications)


def global_vs_local(X, y_bin, y_multi, model_type, reps=30):
    """
    Train and test global model and local model

    :param X: The features
    :type X: pd dataframe
    :param y_bin: Binary labels
    :type y_bin: pd series
    :param y_multi: Multi-class labels
    :type y_multi: pd series
    :param model_type: Type of model used to classify
    :type: model_type: String
    :param reps: Replications
    :type reps: int
    """
    subs = [i for i in range(1, len(y_multi.value_counts()))]
    for sub in subs:
        record_metrics = ['acc', 'bacc', 'prec', 'rec', 'spec', 'f1', 'npv']
        metrics_global = {key: [] for key in record_metrics}
        metrics_local = {key: [] for key in record_metrics}

        # logging setup #
        for rep in range(reps):
            if rep == 0:
                log_file = 'log/global_vs_local/' + DATASET + "_" + model_type + "_" + str(sub) + '.txt'
                sys.stdout = open(log_file, "w")

            # initialize the classifier #
            if model_type == 'XGB':
                global_clf = xgb.XGBClassifier()
                local_clf = xgb.XGBClassifier()
            elif model_type == 'LR':
                global_clf = LogisticRegression(max_iter=10000)
                local_clf = LogisticRegression(max_iter=10000)
            elif model_type == 'DT':
                global_clf = DecisionTreeClassifier()
                local_clf = DecisionTreeClassifier()
            elif model_type == 'RF':
                global_clf = RandomForestClassifier()
                local_clf = RandomForestClassifier()
            else:
                raise Exception("Model type not available.")

            # global train #
            X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, stratify=y_multi,
                                                                random_state=rep)
            global_clf.fit(X_train, y_train)

            # local train #
            X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.3, stratify=y_multi,
                                                                random_state=rep)

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

            if rep == (reps - 1):
                # save output
                filename_global = "results/global_vs_local/" + DATASET + "_" + "global_" + model_type + str(sub) + \
                                  ".csv"
                filename_local = "results/global_vs_local/" + DATASET + "_" + "local_" + model_type + str(sub) + ".csv"
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
