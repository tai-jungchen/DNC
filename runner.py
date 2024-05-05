"""
file name: runner.py
Author: Alex

This class serves as the main function of the project
"""
from tqdm import tqdm
from binary_clf import binary
from dnc import divide_n_conquer
from multi_class_clf import multi_clf, multi_analysis


N_REPS = 10
STRATEGY_1 = "OvO"
STRATEGY_2 = "OvR"
STRATEGY_3 = "Direct"


def main():
	"""
	Carry out the binary vs. OvO vs. DNC comparison
	"""
	models = ['LR', 'DT', 'RF', 'XGBOOST']
	# models = ['LR']
	dataset = 'mnist'

	for model in tqdm(models):
		binary(model, N_REPS, dataset)
		divide_n_conquer(model, N_REPS, dataset)
		multi_clf(model, N_REPS, STRATEGY_1, dataset)
		multi_clf(model, N_REPS, STRATEGY_3, dataset)

		# multi_analysis(model, N_REPS, STRATEGY_1)


if __name__ == "__main__":
	main()
