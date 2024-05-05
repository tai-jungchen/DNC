"""
file name: data_pre.py
Author: Alex

Preprocess the maintenance dataset and NIJ dataset. Specifically, split Power Failure into two categories, and delete
Random Failure
"""
import pandas as pd
from sklearn.decomposition import PCA


def main():
    """
    Call pre-processing functions for the two datasets
    """
    maintenance_pre()
    # nij_pre()
    # mnist()


def mnist(save=False):
    """
    Partition mnist dataset by sample 100 cases from digit 0 - 9, instead of digit 1.

    :param save: Whether save the data to .csv file
    """
    df_train = pd.read_csv('datasets/raw/mnist_train.csv')
    df_test = pd.read_csv('datasets/raw/mnist_test.csv')
    df = pd.concat([df_train, df_test], axis=0)

    # sample only 100 samples for digits other than 1
    unique_labels = df_train['label'].unique()

    sub_samples = []
    for label in unique_labels:
        if label == 0:
            label_samples = df[df['label'] == label].sample(n=6903, replace=True, random_state=42)
        else:
            label_samples = df[df['label'] == label].sample(n=100, replace=True, random_state=42)
        sub_samples.append(label_samples)

    sub_df = pd.concat(sub_samples)

    # move the first column to the last
    first_column = sub_df.iloc[:, 0]
    # Remove the first column from the DataFrame
    sub_df = sub_df.iloc[:, 1:]
    # Concatenate the first column to the end of the DataFrame
    sub_df = pd.concat([sub_df, first_column], axis=1)

    # add one binary classification label
    sub_df['label_bin'] = sub_df['label'].apply(lambda x: 1 if x != 0 else 0)

    if save:
        sub_df.to_csv('mnist_imb.csv', index=False)


def nij_pre(save=False):
    df = pd.read_csv('datasets/Data/NIJ_s_Recidivism_Challenge_Training_Dataset.csv')
    df = df[df['Gender'] == 'F']
    all_year = df['Recidivism_Within_3years'].value_counts()
    year1 = df['Recidivism_Arrest_Year1'].value_counts()
    year2 = df['Recidivism_Arrest_Year2'].value_counts()
    year3 = df['Recidivism_Arrest_Year3'].value_counts()

    # drop year 1 recidivism
    # df = df[df['Recidivism_Arrest_Year1'] == False]
    df['Recidivism_Year'] = df.apply(new_column_value, axis=1)

    data = pd.read_pickle('datasets/Local_F_mean_imputation.pkl')
    # data = data[data['Recidivism_Arrest_Year1'] == False]

    # drop useless feature and old label
    data = data.drop(columns=['ID', 'Gender', 'Recidivism_Arrest_Year1'])
    # add new label columns
    data['Recidivism'] = df['Recidivism_Within_3years']
    data['Recidivism_Year'] = df['Recidivism_Year']

    # save output to .csv file
    if save:
        data.to_csv('nij_data.csv', index=False)


def new_column_value(row):
    """
    Helper function for encoding the label
    :param row: instance
    :return: encoded label
    """
    if row['Recidivism_Arrest_Year1'] is True:
        return 1
    elif row['Recidivism_Arrest_Year2'] is True:
        return 2
    elif row['Recidivism_Arrest_Year3'] is True:
        return 3
    else:
        return 0


def maintenance_pre(save=False):
    """
    Split Power Failure into two categories according to PCA result, and delete Random Failure.
    """
    df = pd.read_csv("datasets/raw/predictive_maintenance.csv")

    # recode failure type
    df = df[df['failure.type'] != 'Random Failures']
    df.loc[df['failure.type'] == 'No Failure', 'failure.type'] = 0
    df.loc[df['failure.type'] == 'Heat Dissipation Failure', 'failure.type'] = 1
    df.loc[df['failure.type'] == 'Power Failure', 'failure.type'] = 2
    df.loc[df['failure.type'] == 'Overstrain Failure', 'failure.type'] = 4
    df.loc[df['failure.type'] == 'Tool Wear Failure', 'failure.type'] = 5
    # recode type
    df.loc[df['type'] == 'H', 'type'] = 1
    df.loc[df['type'] == 'M', 'type'] = 2
    df.loc[df['type'] == 'L', 'type'] = 3

    # PCA
    pca = PCA(n_components=2)

    X = df[df.columns.tolist()[:-2]]
    transformed_data = pca.fit_transform(X)

    df['PCA1'] = transformed_data[:, 0]
    df['PCA2'] = transformed_data[:, 1]

    cond1 = df['PCA1'] >= 500
    cond2 = df['failure.type'] == 2
    df.loc[cond1 & cond2, 'failure.type'] = 3

    # save output to .csv file
    if save:
        df.to_csv('maintenance_data.csv', index=False)


if __name__ == "__main__":
    main()
