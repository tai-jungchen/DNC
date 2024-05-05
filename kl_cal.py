"""
file name: kl_cal.py
Author: Alex

This code plays with the KL divergence on the Machine Predictive Maintenance Classification dataset.
"""
import math
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.special import kl_div
import matplotlib.pyplot as plt
import seaborn as sns

SUBS = [1, 2, 3, 4, 5]
N_REPS = 10
EPSILON = 1e-10


def main():
	"""
	Carry out the KL divergence calculation
	"""
	# settings
	n_reps = N_REPS

	# read data
	df = pd.read_csv("predictive_maintenance.csv")
	df['type'] = df['type'].astype('category').cat.codes
	df['failure.type'] = df['failure.type'].astype('category')
	df['failure.type'].cat.set_categories(['No Failure', 'Heat Dissipation Failure', 'Power Failure',
										   'Overstrain Failure', 'Tool Wear Failure',
										   'Random Failures'], inplace=True)
	df['failure.type'] = df['failure.type'].astype('category').cat.codes

	y = df['failure.type']
	X = df.drop(columns='target')

	for i in range(n_reps):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

		# pass vs. sub-classes
		for sub in SUBS:
			pass_condition = (X_train['failure.type'] == 0)
			fk_condition = (X_train['failure.type'] == sub)

			X_train_fk = X_train[fk_condition].drop(columns='failure.type')
			X_train_pass = X_train[pass_condition].drop(columns='failure.type')

			kl_divergence = kl_calculator(X_train_pass['torque'].to_numpy(), X_train_fk['torque'].to_numpy())
			print(f'KL divergence between pass and failure type {sub}: {kl_divergence}')

		# pass vs. all failure types
		pass_condition = (X_train['failure.type'] == 0)
		fail_condition = (X_train['failure.type'] != 0)
		X_train_fail = X_train[fail_condition].drop(columns='failure.type')
		X_train_pass = X_train[pass_condition].drop(columns='failure.type')

		kl_divergence = kl_calculator(X_train_pass['torque'].to_numpy(), X_train_fail['torque'].to_numpy())
		print(f'KL divergence between pass and all failure types: {kl_divergence}')


def convert_to_dists(p, q):
	"""
	Turn the input features into probability distribution

	:param p: Raw data for the first distribution
	:type p: ndarray

	:param q: Raw data for the second distribution
	:type q: ndarray

	:return: The probability distributions of the given the input data
	:type: a tuple of two lists
	"""
	# Edge case that value is negative
	if (p < 0).any() or (q < 0).any():
		print("Found negative value in the column")
		return None

	# convert to probability distribution
	freq_p = Counter(p)
	freq_q = Counter(q)
	common_support = np.union1d(list(freq_p.keys()), list(freq_q.keys()))

	# align the two distributions
	for i in range(len(common_support)):
		key = common_support[i]
		# zero padding for both distributions
		if key not in freq_p:
			freq_p[key] = EPSILON
		if key not in freq_q:
			freq_q[key] = EPSILON

	freq_p_sum = sum(freq_p.values())
	freq_q_sum = sum(freq_q.values())

	dist_p = {}
	dist_q = {}
	for x in freq_p:
		dist_p[x] = freq_p[x] / freq_p_sum
		dist_q[x] = freq_q[x] / freq_q_sum

	# Sort the dictionary by keys and create a new ordered dictionary
	ordered_p, ordered_q = {key: dist_p[key] for key in sorted(dist_p)}, {key: dist_q[key] for key in sorted(dist_q)}
	return list(ordered_p.values()), list(ordered_q.values())


def kl_calculator(p, q):
	"""
	Calculate the KL divergence between p distribution and q distribution

	:param p: The data of the distribution other than the reference distribution
	:type p: nd array

	:param q: The data of the reference distribution
	:type q: ndarray

	:return: KL(p || q)
	:type: double
	"""
	dist_p, dist_q = convert_to_dists(p, q)

	# Ensure both lists are of the same length
	if len(dist_p) != len(dist_q):
		raise ValueError("Both lists must have the same length")

	# Calculate KL divergence
	kl = 0.0
	for i in range(len(dist_p)):
		if dist_p[i] != 0 and dist_q[i] != 0:  # Avoid log(0)
			kl += dist_p[i] * math.log(dist_p[i] / dist_q[i])
	# print(f'KL divergence by hand: {kl}')
	# print(f'KL divergence by package: {kl_div(dist_p, dist_q).sum()}')
	return kl


if __name__ == "__main__":
	main()
