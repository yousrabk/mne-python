""".

ICASSP 2017
Plot figure 2. : source estimates

"""

import mne
import numpy as np
from mne.inverse_sparse.mxne_optim import (
    iterative_mixed_norm_solver_hyperparam, norm_l2inf,
    iterative_mixed_norm_solver)

import matplotlib
import matplotlib.pyplot as plt
# import scipy.io as sio

# ######## Mixed norm one hp #############
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

noise = 1.4e-2
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
alpha = 0.5 * alpha_max_b  # * np.ones((n_sources),)
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

# ######## iterative Mixed norm one hp #############

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
alpha = 0.5 * alpha_max_b  # * np.ones((n_sources),)
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


# ######## Mixed norm one hp per source #############

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

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
plt.rc('text', usetex=True)

# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True,
#                                            figsize=(8, 6))
# import  matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 7))
fig.subplots_adjust(left=0.12, right=0.97, wspace=0.05, hspace=0.19,
                    bottom=0.08, top=0.88)
X = [(2, 1, 1), (2, 2, 3), (2, 2, 4)]
# for nrows, ncols, plot_number in X:
#     sub = fig.add_subplot(nrows, ncols, plot_number)
#     sub.set_xticks([])
#     sub.set_yticks([])

ax1 = fig.add_subplot(X[0][0], X[0][1], X[0][2])
ax2 = fig.add_subplot(X[1][0], X[1][1], X[1][2])
ax3 = fig.add_subplot(X[2][0], X[2][1], X[2][2])

col = 'lightgreen'
fontsize = 16
# mxne with one hp
ax1.plot(stc.times * 1000, stc.data.T, '*', color=col,
         label='Simulated')
ax1.plot(stc.times * 1000, X_est.T, color=col, label='Esimated')
# ax1.set_xticks([])
ax1.set_xlim([8., 200.])
ax1.set_ylim([-0.1, 1.])
ax1.set_yticks(np.arange(0., 1.1, 0.2))
ax1.set_ylabel('Source Amplitude', fontsize=fontsize)
# ax1.set_title('(a)', loc='left')
ax1.set_title(r'(a) $\ell_{2,1}$ - one hyperparam')
# ax1.set_title('one hyperparam', loc='right')

# irmxne with one hp
ax2.plot(stc.times * 1000, stc.data.T, '*', color=col,
         label='Simulated')
ax2.plot(stc.times * 1000, X_est_i.T, color=col, label='Estimated')
# ax2.set_xticks([])
ax2.set_xlim([8., 200.])
ax2.set_ylim([-0.1, 1.])
ax2.set_yticks(np.arange(0., 1.1, 0.2))
ax2.set_ylabel('Source Amplitude', fontsize=fontsize)
ax2.set_xlabel('Time (ms)', fontsize=fontsize)
# ax2.set_title('(b)', loc='left')
ax2.set_title(r'(b) $\ell_{2,0.5}$ - one hyperparam')
# ax2.set_title('one hyperparam', loc='right')

# mxne with a vector of hp
ax3.plot(stc.times * 1000, stc.data.T, '*', color=col,
         label='Simulated')
ax3.plot(stc.times * 1000, X_est_v.T, color=col, label='Estimated')
ax3.set_xlabel('Time (ms)', fontsize=fontsize)
ax3.set_xlim([8., 200.])
ax3.set_ylim([-0.1, 1.])
ax3.set_yticks(np.arange(0., 1.1, 0.2))
ax3.set_yticklabels([])
# ax3.set_title('(c)', loc='left')
ax3.set_title(r'(c) $\ell_{2,1}$ - hyperparam per source')
# ax3.set_title('hp per source', loc='right')

# # irmxne with a vector of hp
# ax4.plot(stc.times * 1000, stc.data.T / scale, '*', color=col,
#          label='Simulated')
# ax4.plot(stc.times * 1000, X_est_iv.T, color=col, label='Estimated')
# ax4.set_xlabel('Time (ms)', fontsize=fontsize)
# ax4.set_xlim([8., 200.])
# ax4.set_title('(d)', loc='left')
# ax4.set_title(r'\ell_{2,0.5} - ')
# ax4.set_title('hp per source', loc='right')

axes = [ax1, ax2, ax3]
colors = ['c', 'g', 'r', 'lightskyblue']
# datas = [X_est, X_est_i, X_est_v, X_est_iv]
datas = [X_est, X_est_i, X_est_v]
for i, ax in zip(range(len(datas)), axes):
    for j, (data, col) in enumerate(zip(datas[i][:4], colors)):
        ax.plot(stc.times * 1000, stc.data[j], '*', color=col)
        ax.plot(stc.times * 1000, data, color=col)

handles, labels = ax2.get_legend_handles_labels()
plt.legend([handles[0], handles[-1]], [labels[0], labels[-1]],
           bbox_to_anchor=(0.48, 2.45), loc=1, borderaxespad=0., ncol=4)
# plt.tight_layout()
plt.show()
