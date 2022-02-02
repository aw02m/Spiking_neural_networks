from cmath import nan
import numpy as np
import math
import matplotlib.pyplot as plt

print(np.load('bifparams.npy')[35])
bifparams = np.load('bifparams.npy')[:, 2].reshape(74, 119)

# for i in range(bifparams.shape[0]):
for i in range(bifparams.shape[0]):
    for j in range(bifparams.shape[1]):
        if math.isnan(bifparams[i][j]):
            bifparams[i][j] = 3

bifparams = bifparams[:35, :]

plt.imshow(bifparams, cmap='jet', interpolation='none', origin='lower', aspect='auto', extent=[0.01,1.2,0.01,0.36])
plt.colorbar()
# plt.clim()
plt.savefig('gq.jpg')
# plt.show()