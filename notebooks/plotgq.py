from cmath import nan
import numpy as np
import math
import matplotlib.pyplot as plt

# print(np.load('bifparams_first_gq.npy')[35])
# bifparams = np.load('bifparams_first_gq.npy')[:, 2].reshape(74, 119)
# bifparams1 = np.load('bifparams_1.npy')
# bifparams2 = np.load('bifparams_2.npy')
# bifparams3 = np.load('bifparams_3.npy')
bifparams1 = np.load('bifparams_second_1.npy')
bifparams2 = np.load('bifparams_second_2.npy')
bifparams3 = np.load('bifparams_second_3.npy')
bifparams = np.block([[bifparams1],[bifparams2],[bifparams3]])[:, 2].reshape(30, 60)
print(bifparams)

# for i in range(bifparams.shape[0]):
#     for j in range(bifparams.shape[1]):
#         if math.isnan(bifparams[i][j]):
#             bifparams[i][j] = 4

bifparams = bifparams[:, :]

plt.imshow(bifparams, cmap='jet', interpolation='none', origin='lower', aspect='auto', extent=[0.01,1.2,0.0,0.29])
plt.colorbar()
# plt.clim()
plt.savefig('gq_positive_beta.jpg')
# plt.show()