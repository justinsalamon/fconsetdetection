import numpy as np

def get_avg_precision(filename):
	# Get column with precisions - first column in outputs
	P = np.loadtxt(filename)[:,0]
	# Return avg
	return np.mean(P)

filename = '../../results/NSDNS_SF_63_63_new_way.txt'
print get_avg_precision(filename)