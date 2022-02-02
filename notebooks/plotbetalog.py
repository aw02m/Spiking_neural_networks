from cmath import nan
import numpy as np
import math
import matplotlib.pyplot as plt

bifparams = np.load('betalog.npy')[:, 1:3]

plt.plot(bifparams[:, 0], bifparams[:, 1])
plt.savefig('betalog.jpg')