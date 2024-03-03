"""
file name: data_pre.py
Author: Alex

Preprocess the maintenance dataset. Specifically, split Power Failure into two categories, and delete Random Failure
"""
import pandas as pd
from sklearn.decomposition import PCA


def main():
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
	# df.to_csv('maintenance_data.csv', index=False)


if __name__ == "__main__":
	main()
