import mne
import numpy as np
from mne.inverse_sparse.mxne_optim import (dgap_l21, norm_l2inf, prox_l21,
                                           groups_norm2, norm_l21, sum_squared)

stc_fname = 'stc_mind'
stc = mne.read_source_estimate(stc_fname)

tmin, tmax = 0.0085, 0.205
stc.crop(tmin, tmax)
stc._data *= 1e08

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 120, 500, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)

X = 0.1 * rng.randn(n_sources, n_times)
X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)

def solver_prox(M, G, alpha, lipschitz_constant, maxit=200,
                prox_iter=1, update_alpha=False, tol=1e-8, n_orient=1,
                a=1, b=1):
    """Solves L21 inverse problem with proximal iterations and FISTA"""
    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape

    for i_iter in range(prox_iter):
        if n_sources < n_sensors:
            gram = np.dot(G.T, G)
            GTM = np.dot(G.T, M)
        else:
            gram = None

        X = 0.0
        R = M.copy()
        if gram is not None:
            R = np.dot(G.T, R)

        t = 1.0
        Y = np.zeros((n_sources, n_times))  # FISTA aux variable
        E = []  # track cost function

        active_set = np.ones(n_sources, dtype=np.bool)  # start with full AS

        alphas = np.zeros((prox_iter,))
        norms = np.zeros((prox_iter,))
        # X0, active_set_0 = X_start, active_set_start  # store previous values
        # R, Y, active_set = R_start, Y_start, active_set_start
        for i in range(maxit):
            # print("alpha at iter %d: %f" % (i, alpha))
            X0, active_set_0 = X, active_set  # store previous values
            # if (i_iter == 1) and (i == 200):
            #     1/0
            if gram is None:
                Y += np.dot(G.T, R) / lipschitz_constant  # ISTA step
            else:
                Y += R / lipschitz_constant  # ISTA step
            # if i_iter == 3: #alpha.max() > 100:
            X, active_set = prox_l21(Y, alpha / lipschitz_constant, n_orient)

            t0 = t
            t = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t ** 2))
            Y.fill(0.0)
            dt = ((t0 - 1.0) / t)
            Y[active_set] = (1.0 + dt) * X
            Y[active_set_0] -= dt * X0
            Y_as = active_set_0 | active_set

            if gram is None:
                R = M - np.dot(G[:, Y_as], Y[Y_as])
            else:
                R = GTM - np.dot(gram[:, Y_as], Y[Y_as])

            gap, pobj, dobj, _ = dgap_l21(M, G, X, active_set,
                                          min(alpha[active_set]), n_orient)
            # if i_iter == 0:
            E.append(pobj)
            print("pobj : %s -- gap : %s" % (pobj, gap))
            
            # logger.debug("pobj : %s -- gap : %s" % (pobj, gap))
            if gap < tol:
                # print('Convergence reached ! (gap: %s < %s)' % (gap, tol))
                break
        if active_set.sum() == 0:
            raise Exception("No active dipoles found. "
                            "alpha_space is too big.")
        # print("alpha at iter %d: %f" % (i, alpha))

        if update_alpha:
            # alphas[i_iter] = alpha
            alpha[active_set] =  (62. + a) / (g(X) + b)
            # 1/0
            norms[i_iter] = np.sum(g(X))

        # if np.abs(alphas[i_iter] - alphas[i_iter - 1]) < 1e-4:
        #     break

    return X, active_set, E, alpha[active_set], norms


def prox_l21(Y, alpha, n_orient):
    if len(Y) == 0:
        return np.zeros_like(Y), np.zeros((0,), dtype=np.bool)

    n_positions = Y.shape[0] // n_orient

    rows_norm = np.sqrt((Y * Y.conj()).real.reshape(n_positions,
                                                        -1).sum(axis=1))
    # Ensure shrink is >= 0 while avoiding any division by zero
    shrink = np.maximum(1.0 - alpha / np.maximum(rows_norm, alpha), 0.0)
    active_set = shrink > 0.0
    if n_orient > 1:
        active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
        shrink = np.tile(shrink[:, None], [1, n_orient]).ravel()
    Y = Y[active_set]
    Y *= shrink[active_set][:, np.newaxis]
    return Y, active_set



def dgap_l21(M, G, X, active_set, alpha, n_orient):
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    penalty = norm_l21(X, n_orient, copy=True)
    nR2 = sum_squared(R)
    pobj = 0.5 * nR2 + alpha * penalty

    dual_norm = norm_l2inf(np.dot(G.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    dobj = 0.5 * (scaling ** 2) * nR2 + scaling * np.sum(R * GX)
    # dobj = 0.5 * scaling * nR2 + scaling * np.sum(R * GX)

    gap = pobj - dobj
    return gap, pobj, dobj, R


def dgap_l21_list(M, G, X, active_set, alpha, n_orient):
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    alphaX = X * np.tile(alpha[active_set], (X.shape[1], 1)).T
    penalty = norm_l21(alphaX, n_orient, copy=True)
    nR2 = sum_squared(R)
    pobj = 0.5 * nR2 + penalty

    dual_norm = norm_l2inf(np.dot(G.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = np.min([scaling, np.ones((scaling.shape[0],))], axis=0)

    # dobj = 0.5 * (scaling ** 2) * nR2 + scaling * np.sum(R * GX)
    scalingX = X * np.tile(scaling[active_set], (X.shape[1], 1)).T
    GscalingX = np.dot(G[:, active_set], scalingX)
    Rs = M - GscalingX
    dobj = 0.5 * sum_squared(Rs) + np.sum(Rs * GX)

    gap = pobj - dobj
    return gap, pobj, dobj, R


def g(w):
    return np.sqrt(groups_norm2(w.copy(), n_orient))


n_orient = 1
alpha_max = norm_l2inf(np.dot(G.T, M), n_orient)
alpha_max *= 0.01
G /= alpha_max

lc = 1.05 * linalg.norm(G, ord=2) ** 2

mode = alpha_max * 100 / 2.
b = 1.
a = mode * b + 1.

alpha = 1. * np.ones((X.shape[0],))
# alpha = 242.4 * np.ones((X.shape[0],))
# alpha[0] = 10.10
# alpha[1] = 12.58
# alpha[2] = 6.32
# alpha[3] = 5.06
out = solver_prox(M, G, alpha, lc, maxit=10000, tol=1e-6, n_orient=1,
                  prox_iter=5, update_alpha=True, a=a, b=b)

X_est, active_set, E, alphas, norms = out
X_est /= alpha_max

import matplotlib.pyplot as plt
plt.close('all')
# plt.plot(np.log10(E - np.min(E)))
plt.plot(E)
plt.show()

