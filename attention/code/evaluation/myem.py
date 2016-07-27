import numpy as np
import logging
logger = logging.getLogger(__name__)

def myem(data):
	sigma_one = 2
	sigma_two = 2
	a = .5

	while True:
		to_one = a     / np.sqrt(2 * np.pi * sigma_one**2) * np.exp(-data**2     / (2*sigma_one**2))
		to_two = (1-a) / np.sqrt(2 * np.pi * sigma_two**2) * np.exp(-(1-data)**2 / (2*sigma_two**2))
		total = to_one + to_two
		to_one /= total
		to_two /= total

		sigma_one = np.sqrt((to_one * data**2).sum() / to_one.sum())
		sigma_two = np.sqrt((to_two * (1-data)**2).sum() / to_two.sum())
		a = to_one.mean()
		yield sigma_one, sigma_two, a

def fit(data, max_steps=None, min_change=None):
	last = None
	step = 0
	for current in myem(data):
		if min_change is not None and last is not None and (((last-current) / last) <= min_change).all():
			logger.debug("Convergence condition met.") 
			break
		if max_steps is not None and max_steps <= step:
			logger.debug("Maximum number of steps reached.") 
			break

		step += 1
		last = np.array(current)
	
	return current


def log_likelihood(data, sigma_one, sigma_two, a):
	llh_one = a     / np.sqrt(2 * np.pi * sigma_one**2) * np.exp(-data**2     / (2*sigma_one**2))
	llh_two = (1-a) / np.sqrt(2 * np.pi * sigma_two**2) * np.exp(-(1-data)**2 / (2*sigma_two**2))
	return np.sum(np.log(llh_one + llh_two))

def __test__(sigma_one, sigma_two, a, n, num_steps):
	data_one = np.random.normal(loc=0, scale=sigma_one, size=a*n)
	data_two = np.random.normal(loc=1, scale=sigma_two, size=(1-a)*n)
	data = np.concatenate((data_one, data_two))
	for params, _ in zip(myem(data), range(num_steps)):
		print(params)
