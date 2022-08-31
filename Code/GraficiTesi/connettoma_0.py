# Plotta la matrice del connettoma con la colorbar

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


W66=np.loadtxt("../Data/connectivity_matrix/weights.txt")
W998=np.loadtxt("../Data/connectivity_matrix/group_mean_connectivity_matrix_file.txt")

ax = plt.gca()
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)

#plt.figure(figsize=(10,5))
#plt.subplot(1,2,1)
im=plt.imshow(W66, norm=colors.LogNorm(), cmap='bwr')
plt.colorbar(im,)

plt.axvline(x=32.5, color='black')
plt.axhline(y=32.5, color='black')


ax.yaxis.set_minor_locator(mticker.FixedLocator((66.0/4.0, 66/4.0*3)))
ax.yaxis.set_minor_formatter(mticker.FixedFormatter(("RH", "LH")))
ax.xaxis.set_minor_locator(mticker.FixedLocator((66.0/4.0, 66/4.0*3)))
ax.xaxis.set_minor_formatter(mticker.FixedFormatter(("RH", "LH")))

plt.setp(ax.yaxis.get_minorticklabels(), rotation=90, size=15, va="center")
ax.tick_params("y",which="minor",pad=25, left=False)
plt.setp(ax.xaxis.get_minorticklabels(), size=15, va="center")
ax.tick_params("x",which="minor",pad=25, left=False)

#plt.subplot(1,2,2)
#plt.imshow(W998, norm=colors.LogNorm(), cmap='bwr')

#plt.savefig("../Figure/connetoma_66.png")
plt.show()

#####################

im=plt.imshow(W998, norm=colors.LogNorm(), cmap='bwr')
plt.colorbar(im,)

plt.axvline(x=499, color='black')
plt.axhline(y=499, color='black')

ax = plt.gca()

ax.yaxis.set_minor_locator(mticker.FixedLocator((998.0/4.0, 998/4.0*3)))
ax.yaxis.set_minor_formatter(mticker.FixedFormatter(("RH", "LH")))
ax.xaxis.set_minor_locator(mticker.FixedLocator((998.0/4.0, 998/4.0*3)))
ax.xaxis.set_minor_formatter(mticker.FixedFormatter(("RH", "LH")))

plt.setp(ax.yaxis.get_minorticklabels(), rotation=90, size=15, va="center")
ax.tick_params("y",which="minor",pad=25, left=False)
plt.setp(ax.xaxis.get_minorticklabels(), size=15, va="center")
ax.tick_params("x",which="minor",pad=25, left=False)

#plt.subplot(1,2,2)
#plt.imshow(W998, norm=colors.LogNorm(), cmap='bwr')

#plt.savefig("../Figure/connetoma_998.png")
plt.show()

############



fig, axs = plt.subplots(1, 2, constrained_layout=True)


pcm=axs[0].imshow(W66, norm=colors.LogNorm(), cmap='bwr')
axs[0].axvline(x=32.5, color='black')
axs[0].axhline(y=32.5, color='black')
axs[0].set_title("66 ROI")



pcm=axs[1].imshow(W998, norm=colors.LogNorm(), cmap='bwr')
axs[1].axvline(x=499, color='black')
axs[1].axhline(y=499, color='black')
axs[1].set_title("998 ROI")



fig.colorbar(pcm, ax=axs[0:], location='right', shrink=0.5)
plt.show()
