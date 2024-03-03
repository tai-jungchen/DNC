"""
file name: all_vs_sub.py
Author: Alex

Test the performance of majority-minority (all) and majority-minority_sub (sub).
"""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

RAND = 520


def main():
    """
    Carry out the comparison.
    """
    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")

    subs = [1, 2, 3, 4, 5]
    for sub in subs:
        ada_boost_helper(df, 'multi', 'xgBoost', sub=sub)
    ada_boost_helper(df, 'binary', 'xgBoost')


def ada_boost_helper(data, class_type, model_type, sub=1):
    """
    Conduct AdaBoost on the data given the classification type
    :param sub: subclass
    :type sub: int
    :param model_type: type of model used to classify
    :type: model_type: String
    :param data: original data
    :type data: pandas dataframe
    :param class_type: classification type (binary or multiclass)
    :type class_type: String
    """
    if class_type == 'multi':
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-4], data['failure.type'], test_size=0.3,
                                                            stratify=data['failure.type'], random_state=RAND)
        # select only majority and minority sub
        X_train = X_train[(y_train == sub) | (y_train == 0)]
        X_test = X_test[(y_test == sub) | (y_test == 0)]
        y_train = y_train[(y_train == sub) | (y_train == 0)]
        y_test = y_test[(y_test == sub) | (y_test == 0)]

        # turn non-zero sub minority into 1
        y_train[y_train != 0] = 1
        y_test[y_test != 0] = 1

    elif class_type == 'binary':
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-4], data['target'], test_size=0.3,
                                                            stratify=data['failure.type'], random_state=RAND)

    clf = None

    if model_type == 'xgBoost':
        clf = xgb.XGBClassifier()
    elif model_type == 'adaBoost':
        base_classifier = DecisionTreeClassifier()
        clf = AdaBoostClassifier(estimator=base_classifier, random_state=42, n_estimators=50)

    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    # performance
    print("Training:")
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_train, y_pred_train))

    print("Testing:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
