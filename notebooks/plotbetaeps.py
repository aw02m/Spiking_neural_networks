from cmath import nan
import numpy as np
import math
import matplotlib.pyplot as plt

bifparams1 = np.load('betaeps_1.npy')
bifparams2 = np.load('betaeps_2.npy')
bifparams3 = np.load('betaeps_3_.npy')
# bifparams1 = np.load('betaeps_second_1.npy')
# bifparams2 = np.load('betaeps_second_2.npy')
# bifparams3 = np.load('betaeps_second_3.npy')

bifparams = np.block([[bifparams1],[bifparams2],[bifparams3]])[:, 2].reshape(31, 31)

bifparams = bifparams[:, :]

plt.imshow(bifparams, cmap='jet', interpolation='none', origin='lower', aspect='auto', extent=[0.025, 0.1, 0.01, 0.04])
plt.xlabel("ε")
plt.ylabel("β")
plt.colorbar()
# plt.clim()
plt.savefig('betaeps_.jpg')
# plt.show()