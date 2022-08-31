import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl

import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable



centers=np.genfromtxt("../Data/connectivity_matrix/centres.txt",dtype=None, usecols=(1,2,3))
labels=np.genfromtxt("../Data/connectivity_matrix/centres.txt",dtype=str, usecols=(0))

ax = plt.gca()
#fig = plt.figure(figsize=(5,5), dpi=100)
#ax = fig.add_subplot()#projection='3d')
centers=centers.transpose()
for i,label in enumerate(labels):
    plt.scatter(centers[0,i],centers[1,i], marker='o', color='blue')
    #print(label)
    plt.text(centers[0,i]+2,centers[1,i]+2, label, size=9)

ax.yaxis.set_minor_locator(mticker.FixedLocator((130, 50)))
ax.yaxis.set_minor_formatter(mticker.FixedFormatter(("RH", "LH")))
ax.xaxis.set_minor_locator(mticker.FixedLocator((70, 170)))
ax.xaxis.set_minor_formatter(mticker.FixedFormatter(("Posteriore", "Anteriore")))


plt.setp(ax.yaxis.get_minorticklabels(), rotation=90, size=15, va="center")
ax.tick_params("y",which="minor",pad=25, left=False)
plt.setp(ax.xaxis.get_minorticklabels(), size=15, va="center")
ax.tick_params("x",which="minor",pad=25, left=False)

plt.axline()

#ax.set_title("3D plot delle posizioni dei ROI")
plt.show()


