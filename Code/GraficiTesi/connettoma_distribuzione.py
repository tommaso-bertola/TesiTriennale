# Mostra la distribuzione dei pesi dei connettomi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

W66 = np.loadtxt("../Data/connectivity_matrix/weights.txt")
W998 = np.loadtxt(
    "../Data/connectivity_matrix/group_mean_connectivity_matrix_file.txt")

w66_0=W66[W66!=0].flatten()
w998_0=W998[W998!=0].flatten()

#plt.figure(figsize=(10, 5))
#plt.subplot(1, 2, 1)
plt.hist(w66_0, log=True, bins=50, range=(0, 0.6), density=True,
         histtype='step', label='66 ROI')
plt.hist(w998_0, log=True, bins=50, range=(0, 0.6), density=True,
         histtype='step', label='998 ROI')
plt.legend()
#plt.title('Distribuzione dei pesi per i due connettomi')

plt.xlabel("Peso")
plt.ylabel("Percentuale di occorrenza")

#plt.subplot(1, 2, 2)
#plt.hist(w66_reduced, log=True, bins=50, range=(0, 0.6), density=True,
#         histtype='step', label='w66_r vmax=%.2f' % w66_reduced.max())
#plt.hist(w998_reduced, log=True, bins=50, range=(0, 0.6), density=True,
#         histtype='step', label='w998_r vmax=%.2f' % w998_reduced.max())
#plt.legend()
#plt.title('Distribution of weights\nreduced connectome')

plt.savefig("../Figure/distribuzione_pesi_connettomi.png")
plt.show()
