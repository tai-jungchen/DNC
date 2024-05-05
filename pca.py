"""
file name: pca.py
Author: Alex

Conduct PCA on the dataset and visualize the data after dimension reduction
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main():
	"""
	Visualize the PCA result under binary and multiclass scenario.
	"""
	df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")

	pca_helper(df[df.columns.tolist()[:-4]], df['failure.type'])
	pca_helper(df[df.columns.tolist()[:-4]], df['target'])


def pca_helper(X, y):
	"""
	Conduct PCA on the data and plot out the data after dimension reduction
	:param data: original data
	:type data: pandas dataframe
	:param class_type: classification type (binary or multiclass)
	:type class_type: String
	"""
	# multi-class
	if len(y.unique()) > 2:
		# Instantiate PCA object
		pca = PCA()

		# Fit and transform the data
		transformed_data = pca.fit_transform(X)

		# Plot the transformed data with different colors for different labels
		plt.scatter(transformed_data[y == 0, 0], transformed_data[y == 0, 1], c='blue', label='Pass')
		plt.scatter(transformed_data[y == 1, 0], transformed_data[y == 1, 1], c='red', label='Failure type 1')
		plt.scatter(transformed_data[y == 2, 0], transformed_data[y == 2, 1], c='green', label='Failure type 2')
		plt.scatter(transformed_data[y == 3, 0], transformed_data[y == 3, 1], c='orange', label='Failure type 3')
		plt.scatter(transformed_data[y == 4, 0], transformed_data[y == 4, 1], c='cyan', label='Failure type 4')
		plt.scatter(transformed_data[y == 5, 0], transformed_data[y == 5, 1], c='magenta', label='Failure type 5')

		# Add labels and legend
		plt.title('PCA with Different Colored Labels')
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.legend()
		plt.show()
	elif len(y.unique()) == 2:
		# Instantiate PCA object
		pca = PCA()

		# Fit and transform the data
		transformed_data = pca.fit_transform(X)

		# Plot the transformed data with different colors for different labels
		plt.scatter(transformed_data[y == 0, 0], transformed_data[y == 0, 1], c='blue', label='Pass')
		plt.scatter(transformed_data[y == 1, 0], transformed_data[y == 1, 1], c='red', label='Failure')

		# Add labels and legend
		plt.title('PCA with Different Colored Labels')
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.legend()
		plt.show()
	else:
		raise Exception("Invalid number of labels")


if __name__ == "__main__":
	main()
