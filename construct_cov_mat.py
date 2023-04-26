"""
Uses a fit result to construct a covariance matrix 
"""

import json
import os
import matplotlib.pyplot as plt 
import numpy as np

from rotations import construct_rotation

from math import sin, cos, sqrt

state_file = "ellipse_state.json"

DIM = 9


_obj = open(state_file, 'rt')
data  = json.load(_obj)
_obj.close()
x0 = data["params"]

eigenvals = 1.0/np.array(x0[0:DIM])

thetas = x0[DIM:]

rotation_mat =  np.transpose(construct_rotation(thetas, DIM))


# check rotation matrix is in SO(9)
cross_check = np.matmul(rotation_mat, np.transpose(rotation_mat))
for i in range(DIM):
    for j in range(DIM):
        if i==j:
            if not abs(cross_check[i][j]-1)<1e-15:
                raise ValueError("Found non-1 diagonal at {}x{}, {}".format(i, j, cross_check[i][j]))
        else:
            if not abs(cross_check[i][j])<1e-15:
                raise ValueError("Found non-zero off-diagonal at {}x{}, {}".format(i, j, cross_check[i][j]))


a_matrix = np.diag(eigenvals)

sigma = np.matmul(rotation_mat, np.matmul(a_matrix, np.linalg.inv(rotation_mat)))
sigma = np.linalg.inv(sigma)

# remove scaling from covariance matrix since it was scaled up
# this gives a correlation matrix
# can use the on-axis widths to re-construct a covariance matrix 
for i in range(9):
    for j in range(9):
        if i == j:
            continue
        sigma[i][j] = sigma[i][j]/sqrt(sigma[i][i]*sigma[j][j])

for i in range(9):
    sigma[i][i]/=sigma[i][i]

# this will raise an exception if `sigma` is not positive-definite (sanity check)
np.linalg.cholesky(sigma)

dimdict = {
    'Amp00'   :     0,
    "Amp01"   :     1,
    "Amp02"   :     2,
    "Amp03"   :     3,
    "Amp04"   :     4,
    "Phase01"   :     5,
    "Phase02"   :     6,
    "Phase03"   :     7,
    "Phase04"   :     8
}

fig, axes = plt.subplots()
mesh = axes.pcolormesh(range(9), range(9), sigma, vmin=-1, vmax=1, cmap='RdBu')
axes.set_xticks(range(9), labels=dimdict.keys(), rotation=45)
axes.set_yticks(range(9), labels=dimdict.keys())
axes.hlines(y=4.5, xmin=-0.5, xmax=8.5, colors='k', linestyles='--')
axes.vlines(x=4.5, ymin=-0.5, ymax=8.5, colors='k', linestyles='--')
plt.colorbar(mesh)
plt.tight_layout()
plt.savefig("./plots/covar.png",dpi=400)



keylist = list(dimdict.keys())
outdict = {}
# now we build the dictionary (yay)
for i in range(9):
    outdict[keylist[i].lower()] = {}
    for j in range(9):
        outdict[keylist[i].lower()][keylist[j].lower()] = sigma[i][j]

print("Correlation matrix determinant {}".format(np.linalg.det(sigma)))

_obj= open(os.path.join(os.path.dirname(__file__), "ice_covariance.json"), 'wt')
json.dump(outdict, _obj, indent=4)
_obj.close()
