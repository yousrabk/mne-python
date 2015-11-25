"""
================================
Compute Rap-Music on evoked data
================================

Compute a Recursively Applied and Projected MUltiple Signal Classification
(RAP-MUSIC) on evoked dataset.

The reference for Rap-Music is:
J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
applied and projected (RAP) MUSIC. Trans. Sig. Proc. 47, 2
(February 1999), 332-340.
DOI=10.1109/78.740118 http://dx.doi.org/10.1109/78.740118
"""

# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt

from mne.beamformer._rap_music import _apply_rap_music


n_orient = 3
n_points = 200
n_sensors = 20
n_times = 50

G = np.random.multivariate_normal(np.zeros(n_points * n_orient),
                                  np.eye(n_points * n_orient), n_sensors)

active_idx = np.array([25, 75, 125, 175])
active_set = np.zeros(n_points * n_orient, dtype=np.bool)
X_sim = np.zeros((n_points * n_orient, n_times))
times = np.linspace(0, 1., num=n_times, endpoint=False)
activation = np.array([np.sin(2 * np.pi * 1. * times) * 100.,
                       np.sin(2 * np.pi * 1. * times + 0.) * 70.,
                       np.sin(2 * np.pi * 3. * times) * 40.,
                       np.sin(2 * np.pi * 6. * times) * 40.])
X_sim[n_orient * active_idx[0]: n_orient * (active_idx[0] + 1)] = \
    np.repeat(activation[0][None, :], n_orient, axis=0)
X_sim[n_orient * active_idx[1]: n_orient * (active_idx[1] + 1)] = \
    np.repeat(activation[1][None, :], n_orient, axis=0)
X_sim[n_orient * active_idx[2]: n_orient * (active_idx[2] + 1)] = \
    np.repeat(activation[2][None, :], n_orient, axis=0)
X_sim[n_orient * active_idx[3]: n_orient * (active_idx[3] + 1)] = \
    np.repeat(activation[3][None, :], n_orient, axis=0)
active_set[n_orient * active_idx[0]: n_orient * (active_idx[0] + 1)] = True
active_set[n_orient * active_idx[1]: n_orient * (active_idx[1] + 1)] = True
active_set[n_orient * active_idx[2]: n_orient * (active_idx[2] + 1)] = True
active_set[n_orient * active_idx[3]: n_orient * (active_idx[3] + 1)] = True
M_GT = np.dot(G, X_sim)

noise = np.random.randn(n_sensors, n_times)

M = M_GT + noise

sol, source_idx_final, oris_final, max_corr = _apply_rap_music(M, G, n_orient,
                                                               n_dipoles=4)

ori_mat = linalg.block_diag(*oris_final).T
X = np.dot(ori_mat, sol)

plt.figure(), plt.plot(X_sim[active_set].T)
plt.figure(), plt.plot(X.T)
plt.show()
