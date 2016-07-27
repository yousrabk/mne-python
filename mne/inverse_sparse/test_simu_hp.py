
import mne
import numpy as np
from mne.inverse_sparse.mxne_optim import (
    iterative_mixed_norm_solver_hyperparam, norm_l2inf,
    iterative_mixed_norm_solver)

import matplotlib.pyplot as plt
import scipy.io as sio

stc_fname = 'stc_mind'
stc = mne.read_source_estimate(stc_fname)

tmin, tmax = 0.0085, 0.205
stc.crop(tmin, tmax)
stc._data *= 1e7

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 304, 22647, stc.data.shape[1]
n_sensors, n_sources, n_times = 102, 300, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)
G *= 10
# G = sio.loadmat('G_whitened.mat')['G']
# G /= G.std(axis=0) 

X = 0.1e-2 * rng.randn(n_sources, n_times)
X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

n_orient = 3
alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
# G /= alpha_max

# ###########
# G = sio.loadmat('G_whitened.mat')['G']
# alpha_max = sio.loadmat('alpha_max.mat')['alpha_max'][0][0]
# # G *= alpha_max
# M = sio.loadmat('M_whitened.mat')['M']

alpha = 60. * alpha_max / 100.
n_mxne_iter = 10
update_alpha = True

if update_alpha:
    solver = iterative_mixed_norm_solver_hyperparam
else:
    solver = iterative_mixed_norm_solver

out = solver(M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-4,
    n_orient=1)

X_est, active_set, E, alphas = out
# X_est /= alpha_max

# plt.close('all')
# plt.plot(np.log10(E - np.min(E)))
# plt.plot(E)
plt.figure()
plt.plot(X_est.T)
plt.show()
