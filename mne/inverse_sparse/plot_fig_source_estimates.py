
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
n_sensors, n_sources, n_times = 102, 500, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)
n_orient = 1

noise = 1.5e-2
E = noise * rng.randn(n_sources, n_times)
X = np.zeros((n_sources, n_times)) + E
X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

scale = 1.
G *= scale

alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
G /= alpha_max

alpha_max_b = norm_l2inf(np.dot(G.T, M), n_orient)
alpha = 0.5 * alpha_max_b #* np.ones((n_sources),)
n_mxne_iter = 1
update_alpha = True

b = 1.  # 1. / scale
a = alpha_max_b / 2. * b + 1
if update_alpha:
    solver = iterative_mixed_norm_solver_hyperparam
else:
    solver = iterative_mixed_norm_solver

out = solver(M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-4,
    n_orient=n_orient, a=a, b=b, hp_iter=10)

X_est, active_set, E, alphas = out
X_est /= alpha_max

# ################################

stc_fname = 'stc_mind'
stc = mne.read_source_estimate(stc_fname)

tmin, tmax = 0.0085, 0.205
stc.crop(tmin, tmax)
stc._data *= 1e7

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 304, 22647, stc.data.shape[1]
n_sensors, n_sources, n_times = 102, 500, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)
n_orient = 1

X = noise * rng.randn(n_sources, n_times)
X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

scale = 1.
G *= scale

alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
G /= alpha_max

alpha_max_b = norm_l2inf(np.dot(G.T, M), n_orient)
alpha = 0.5 * alpha_max_b #* np.ones((n_sources),)
n_mxne_iter = 10
update_alpha = True

b = 1.  # 1. / scale
a = alpha_max_b / 2. * b + 1
if update_alpha:
    solver = iterative_mixed_norm_solver_hyperparam
else:
    solver = iterative_mixed_norm_solver

out = solver(M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-4,
    n_orient=n_orient, a=a, b=b, hp_iter=10)

X_est_i, active_set, E, alphas = out
X_est_i /= alpha_max


# ################################

stc_fname = 'stc_mind'
stc = mne.read_source_estimate(stc_fname)

tmin, tmax = 0.0085, 0.205
stc.crop(tmin, tmax)
stc._data *= 1e7

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 304, 22647, stc.data.shape[1]
n_sensors, n_sources, n_times = 102, 500, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)
n_orient = 1

X = noise * rng.randn(n_sources, n_times)
X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

scale = 1.
G *= scale

alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
G /= alpha_max

alpha_max_b = norm_l2inf(np.dot(G.T, M), n_orient)
alpha = 0.2 * alpha_max_b * np.ones((n_sources),)
n_mxne_iter = 1
update_alpha = True

b = 1.  # 1. / scale
a = alpha_max_b / 2. * b + 1
if update_alpha:
    solver = iterative_mixed_norm_solver_hyperparam
else:
    solver = iterative_mixed_norm_solver

out = solver(M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-4,
    n_orient=n_orient, a=a, b=b, hp_iter=10)

X_est_v, active_set, E, alphas = out
X_est_v /= alpha_max


# ################################

import matplotlib
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16)
plt.rc('text', usetex=True)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True,
                                    figsize=(8, 6))
f.subplots_adjust(left=0.08, right=0.97, wspace=0.05, hspace=0.13,
				  bottom=0.09, top=0.88)
col = 'lightgreen'
fontsize = 16
# mxne
ax1.plot(stc.times * 1000, stc.data.T / scale, '*', color=col, label='Simulated')
ax1.plot(stc.times * 1000, X_est.T, color=col, label='Esimated')
ax1.set_xticks([])
ax1.set_xlim([8., 200.])
ax1.set_yticks(arange(-0.2, 1., 0.2))
ax1.set_ylabel('Simulated Amplitude', fontsize=fontsize)
ax1.set_title('(a)', loc='left')
ax1.set_title(r'\ell_{2,1} - ')
ax1.set_title('one hyperparam', loc='right')

# irmxne
ax2.plot(stc.times * 1000, stc.data.T / scale, '*', color=col, label='Simulated')
ax2.plot(stc.times * 1000, X_est_i.T, color=col, label='Estimated')
ax2.set_xticks([])
ax2.set_xlim([8., 200.])
ax2.set_title('(b)', loc='left')
ax2.set_title(r'\ell_{2,0.5} - ')
ax2.set_title('one hyperparam', loc='right')

# mxne with a vector of hp
ax3.plot(stc.times * 1000, stc.data.T / scale, '*', color=col, label='Simulated')
ax3.plot(stc.times * 1000, X_est_v.T, color=col, label='Estimated')
ax3.set_xlabel('Time (ms)', fontsize=fontsize)
ax3.set_ylabel('Simulated Amplitude', fontsize=fontsize)
ax3.set_xlim([8., 200.])
ax3.set_yticks(arange(-0.2, 1., 0.2))
ax3.set_title('(c)', loc='left')
ax3.set_title(r'\ell_{2,1} - ')
ax3.set_title('hp per source', loc='right')

# irmxne with a vector of hp
ax4.plot(stc.times * 1000, stc.data.T / scale, '*', color=col, label='Simulated')
ax4.plot(stc.times * 1000, X_est_iv.T, color=col, label='Estimated')
ax4.set_xlabel('Time (ms)', fontsize=fontsize)
ax4.set_xlim([8., 200.])
ax4.set_title('(d)', loc='left')
ax4.set_title(r'\ell_{2,0.5} - ')
ax4.set_title('hp per source', loc='right')

axes = [ax1, ax2, ax3, ax4]
colors = ['c', 'g', 'r', 'm']
datas = [X_est, X_est_i, X_est_v, X_est_iv]
for i, ax in zip(range(len(datas)), axes):
	for j, (data, col) in enumerate(zip(datas[i][:4], colors)):
		ax.plot(stc.times * 1000, stc.data[j], '*', color=col)
		ax.plot(stc.times * 1000, data, color=col)

handles, labels = ax2.get_legend_handles_labels()
plt.legend([handles[0], handles[-1]], [labels[0], labels[-1]],
           bbox_to_anchor=(0.5, 2.425), loc=1, borderaxespad=0., ncol=4)
# plt.tight_layout()
plt.show()

