"""
file name: global_vs_local_ana.py
Author: Alex

Analyze why the precision and specificity increase when using local model rather than global model. The model is
trained on dimension reduced data and the decision boundary is plot on the x-y coordinates where the first two
principal components are the x and y axes. Note that the classifier is trained on feature after conducting dimension
reduction, so the result might be slightly different from classifier that is trained on original feature.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb


def main_global(clf, subclass=1):
    """
    Plot the decision boundary and train/test points on the first two PC.

    :param clf: Classifier
    :type clf: String
    :param subclass: The subclass
    :type subclass: int
    """
    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    X = df.iloc[:, :-4]
    y_bin = df['target']
    y_multi = df['failure.type']

    X_train, _, y_train, _ = train_test_split(X, y_bin, stratify=y_multi, test_size=0.3, random_state=0)
    _, X_test, _, y_test = train_test_split(X, y_multi, stratify=y_multi, test_size=0.3, random_state=0)

    # sub-data for testing #
    X_test = X_test[(y_test == sub) | (y_test == 0)]
    y_test = y_test[(y_test == sub) | (y_test == 0)]
    y_test[y_test != 0] = 1

    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Classifier #
    if clf == 'XGB':
        model = xgb.XGBClassifier(seed=42)
    elif clf == 'LR':
        model = LogisticRegression()
    elif clf == 'DT':
        model = DecisionTreeClassifier(random_state=42)
    elif clf == 'RF':
        model = RandomForestClassifier(random_state=42)
    else:
        raise Exception("Model type not available.")
    model.fit(X_train_pca, y_train)

    # take the largest range #
    x_train_min, x_train_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_train_min, y_train_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    x_test_min, x_test_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_test_min, y_test_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    x_min, x_max = min(x_train_min, x_test_min), max(x_train_max, x_test_max)
    y_min, y_max = min(y_train_min, y_test_min), max(y_train_max, y_test_max)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # training
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    for i in range(2):
        plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], label=str(i), s=20, edgecolor='k')
    plt.title('Decision Boundaries for global ' + clf + ' on MPMC sub-data ' + str(subclass) + "(training)")
    plt.xlabel('Principal Component 1 (rot.speed)')
    plt.ylabel('Principal Component 2 (tool.wear)')
    plt.colorbar(label="Predicted Probability")
    plt.legend(title='Failure')
    plt.show()

    # testing
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    for i in range(2):
        plt.scatter(X_test_pca[y_test == i, 0], X_test_pca[y_test == i, 1], label=str(i), s=20, edgecolor='k')
    plt.title('Decision Boundaries for global ' + clf + ' on MPMC sub-data ' + str(subclass) + "(testing)")
    plt.xlabel('Principal Component 1 (rot.speed)')
    plt.ylabel('Principal Component 2 (tool.wear)')
    plt.colorbar(label="Predicted Probability")
    plt.legend(title='Failure')
    plt.show()


def main_local(clf, subclass=1):
    """
    Plot the decision boundary and train/test points on the first two PC.

    :param clf: Classifier
    :type clf: String
    :param subclass: The subclass
    :type subclass: int
    """
    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    X = df.iloc[:, :-4]
    y_multi = df['failure.type']

    X_train, X_test, y_train, y_test = train_test_split(X, y_multi, stratify=y_multi, test_size=0.3, random_state=0)
    # select only majority and minority sub
    X_train = X_train[(y_train == sub) | (y_train == 0)]
    y_train = y_train[(y_train == sub) | (y_train == 0)]
    y_train[y_train != 0] = 1  # turn non-zero sub minority into 1

    # sub dataset for testing #
    X_test = X_test[(y_test == sub) | (y_test == 0)]
    y_test = y_test[(y_test == sub) | (y_test == 0)]
    y_test[y_test != 0] = 1

    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Classifier #
    if clf == 'XGB':
        model = xgb.XGBClassifier(seed=42)
    elif clf == 'LR':
        model = LogisticRegression()
    elif clf == 'DT':
        model = DecisionTreeClassifier(random_state=42)
    elif clf == 'RF':
        model = RandomForestClassifier(random_state=42)
    else:
        raise Exception("Model type not available.")
    model.fit(X_train_pca, y_train)

    # take the largest range #
    x_train_min, x_train_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_train_min, y_train_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    x_test_min, x_test_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_test_min, y_test_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    x_min, x_max = min(x_train_min, x_test_min), max(x_train_max, x_test_max)
    y_min, y_max = min(y_train_min, y_test_min), max(y_train_max, y_test_max)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # training
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    for i in range(2):
        plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], label=str(i), s=20, edgecolor='k')
    plt.title('Decision Boundaries for local ' + clf + ' on MPMC sub-data ' + str(subclass) + "(training)")
    plt.xlabel('Principal Component 1 (rot.speed)')
    plt.ylabel('Principal Component 2 (tool.wear)')
    plt.colorbar(label="Predicted Probability")
    plt.legend(title='Failure')
    plt.show()

    # testing
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    for i in range(2):
        plt.scatter(X_test_pca[y_test == i, 0], X_test_pca[y_test == i, 1], label=str(i), s=20, edgecolor='k')
    plt.title('Decision Boundaries for local ' + clf + ' on MPMC sub-data ' + str(subclass) + "(testing)")
    plt.xlabel('Principal Component 1 (rot.speed)')
    plt.ylabel('Principal Component 2 (tool.wear)')
    plt.colorbar(label="Predicted Probability")
    plt.legend(title='Failure')
    plt.show()


if __name__ == "__main__":
    # clfs = ["LR", "DT", "RF", "XGB"]
    clfs = ["XGB"]
    subs = [1, 2, 3, 4, 5]

    for classifier in clfs:
        for sub in subs:
            main_global(classifier, subclass=sub)
            main_local(classifier, subclass=sub)
