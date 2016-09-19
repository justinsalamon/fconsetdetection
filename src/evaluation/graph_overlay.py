import seaborn
import numpy as np
import matplotlib.pyplot as plt

directory = '../../results/'
files = ['NSDNS_SF_70_70_new_way.txt','NSDNS_SF_64_64_new_way.txt','NSDNS_SF_51_test_51_new_way.txt','NSDNS_20110902_192900_streaming_prob_72_new_way.txt','NSDNS_SF_65_65_new_way.txt']

Ps = [[],[],[],[],[]]
Rs = [[],[],[],[],[]]

for i in xrange(5):
	d = np.loadtxt(directory+files[i])
	Ps[i] = d[:,0]
	Rs[i] = d[:,1]

plt.plot(Rs[3],Ps[3], label='SKM')
plt.plot(Rs[0],Ps[0], label='Weighted Truncated SF')
plt.plot(Rs[1],Ps[1], label='Truncated SF')
plt.plot(Rs[2],Ps[2], label='Pure SF')
plt.plot(Rs[4],Ps[4], label='Baseline')
plt.legend()
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Comparison of Detection Algorithms')
plt.show()