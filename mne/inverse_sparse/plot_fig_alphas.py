
import mne
import numpy as np
from mne.inverse_sparse.mxne_optim import (
    iterative_mixed_norm_solver_hyperparam, norm_l2inf,
    iterative_mixed_norm_solver)

import matplotlib.pyplot as plt

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
n_orient = 1

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 304, 22647, stc.data.shape[1]
n_sensors, n_sources, n_times = 102, 300, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)
n_orient = 1

low_noise = 0.1e-2
high_noise = 3.e-2

X = low_noise * rng.randn(n_sources, n_times)

X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
G /= alpha_max

n_mxne_iter = 1
update_alpha = True
alpha_max_b = norm_l2inf(np.dot(G.T, M), n_orient)

# alphas_init = np.arange(0.01, 0.8, 0.04)
alphas_init = np.arange(0.015, 0.9, 0.2)
rmse = np.zeros((len(alphas_init),))
alphas = {}
as_ = np.zeros((len(alphas_init),))
for i_n, al in enumerate(alphas_init):
    alpha = al * alpha_max_b

    b = 1.  # 1. / scale
    a = alpha_max_b / 2. * b + 1
    if update_alpha:
        solver = iterative_mixed_norm_solver_hyperparam
    else:
        solver = iterative_mixed_norm_solver

    out = solver(M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-4,
                 n_orient=n_orient, a=a, b=b, hp_iter=10)

    X_est, active_set, E, als = out
    X_est /= alpha_max

    rmse[i_n] = np.linalg.norm(stc.data - X_est[:4])
    alphas[i_n] = als
    as_[i_n] = active_set.sum()

# #############################

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
n_orient = 1

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 304, 22647, stc.data.shape[1]
n_sensors, n_sources, n_times = 102, 300, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)
n_orient = 1

X = low_noise * rng.randn(n_sources, n_times)

X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
G /= alpha_max

n_mxne_iter = 10
update_alpha = True
alpha_max_b = norm_l2inf(np.dot(G.T, M), n_orient)

# alphas_init = np.arange(0.015, 0.8, 0.05)
rmse_i = np.zeros((len(alphas_init),))
alphas_i = {}
as_i = np.zeros((len(alphas_init),))
for i_n, al in enumerate(alphas_init):
    alpha = al * alpha_max_b

    b = 1.  # 1. / scale
    a = alpha_max_b / 2. * b + 1
    if update_alpha:
        solver = iterative_mixed_norm_solver_hyperparam
    else:
        solver = iterative_mixed_norm_solver

    out = solver(M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-4,
                 n_orient=n_orient, a=a, b=b, hp_iter=10)

    X_est, active_set, E, als = out
    X_est /= alpha_max

    rmse_i[i_n] = np.linalg.norm(stc.data - X_est[:4])
    alphas_i[i_n] = als
    as_i[i_n] = active_set.sum()

# ############################
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
n_orient = 1

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 304, 22647, stc.data.shape[1]
n_sensors, n_sources, n_times = 102, 300, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)
n_orient = 1

X = high_noise * rng.randn(n_sources, n_times)

X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
G /= alpha_max

n_mxne_iter = 1
update_alpha = True
alpha_max_b = norm_l2inf(np.dot(G.T, M), n_orient)

# alphas_init = np.arange(0.01, 0.8, 0.04)
# alphas_init = np.arange(0.015, 0.8, 0.05)
rmse_noise = np.zeros((len(alphas_init),))
alphas_noise = {}
as_noise = np.zeros((len(alphas_init),))
for i_n, al in enumerate(alphas_init):
	alpha = al * alpha_max_b

	b = 1.  # 1. / scale
	a = alpha_max_b / 2. * b + 1
	if update_alpha:
	    solver = iterative_mixed_norm_solver_hyperparam
	else:
	    solver = iterative_mixed_norm_solver

	out = solver(M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-4,
	    n_orient=n_orient, a=a, b=b, hp_iter=10)

	X_est, active_set, E, als = out
	X_est /= alpha_max

	rmse_noise[i_n] = np.linalg.norm(stc.data - X_est[:4])
	alphas_noise[i_n] = als
	as_noise[i_n] = active_set.sum()

# #############################

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
n_orient = 1

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 304, 22647, stc.data.shape[1]
n_sensors, n_sources, n_times = 102, 300, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)
n_orient = 1

X = high_noise * rng.randn(n_sources, n_times)

X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
G /= alpha_max

n_mxne_iter = 10
update_alpha = True
alpha_max_b = norm_l2inf(np.dot(G.T, M), n_orient)

# alphas_init = np.arange(0.015, 0.8, 0.05)
rmse_inoise = np.zeros((len(alphas_init),))
alphas_inoise = {}
as_inoise = np.zeros((len(alphas_init),))
for i_n, al in enumerate(alphas_init):
	alpha = al * alpha_max_b

	b = 1.  # 1. / scale
	a = alpha_max_b / 2. * b + 1
	if update_alpha:
	    solver = iterative_mixed_norm_solver_hyperparam
	else:
	    solver = iterative_mixed_norm_solver

	out = solver(M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-4,
	    n_orient=n_orient, a=a, b=b, hp_iter=10)

	X_est, active_set, E, als = out
	X_est /= alpha_max

	rmse_inoise[i_n] = np.linalg.norm(stc.data - X_est[:4])
	alphas_inoise[i_n] = als
	as_inoise[i_n] = active_set.sum()


# #############################
import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
f.subplots_adjust(left=0.11, right=0.97, wspace=0.06, hspace=0.15,
                  bottom=0.15)


ax1.axhline(0, linestyle='-', color='k', label='High SNR')
ax1.axhline(0, linestyle='--', color='k', label='Low SNR')
ax1.set_ylim([1e-1, 1e2])

ax2.axhline(0, linestyle='-', color='k', label='High SNR')
ax2.axhline(0, linestyle='--', color='k', label='Low SNR')
ax2.set_ylim([1e-1, 1e2])

ax1.legend(['High SNR', 'Low SNR'], fontsize=14)
ax2.legend(['High SNR', 'Low SNR'], fontsize=14)

for i_d in range(len(alphas)):
    ax1.plot(range(len(alphas[i_d])), alphas[i_d])
    ax1.plot(range(len(alphas_noise[i_d])), alphas_noise[i_d], '--')

for i_d in range(len(alphas)):
    ax2.plot(range(len(alphas_i[i_d])), alphas_i[i_d])
    ax2.plot(range(len(alphas_inoise[i_d])), alphas_inoise[i_d], '--')

# ax1.set_title(r'\|X\|_{2,1}')
ax1.set_title('l_21')
ax1.set_title('(a)', loc='left')
# ax2.set_title(r'\|X\|_{2,0.5}')
ax2.set_title('l_2,0.5')
ax2.set_title('(b)', loc='left')
ax1.set_xticks(range(len(alphas[i_d])))
ax2.set_xticks(range(len(alphas[i_d])))
ax1.set_yscale('log')
# ax1.set_yticks([0, 10, 100])
# ax2.set_yticks([])
ax2.set_xlim([0, 6.5])
ax2.set_xticks(np.arange(7))
ax1.set_xlim([0, 5.5])
ax1.set_xticks(np.arange(6))
ax1.set_xlabel('number of iterations', fontsize=16)
ax2.set_xlabel('number of iterations', fontsize=16)
ax1.set_ylabel('initialisation of lambda', fontsize=16)
# # plt.legend(alphas_init * 100, bbox_to_anchor=(1., 1),
# # 		   loc=1, borderaxespad=0., ncol=4)
# # plt.title(r'convergence of \lambda', fontsize=16)
plt.show()
