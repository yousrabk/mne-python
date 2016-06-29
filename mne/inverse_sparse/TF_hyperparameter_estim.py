
import mne
import numpy as np

from mne.inverse_sparse.mxne_optim import norm_l21_tf, norm_l1_tf
from mne.time_frequency.stft import stft, istft
import matplotlib.pyplot as plt

stc_fname = 'stc_mind'
stc = mne.read_source_estimate(stc_fname)

tmin, tmax = 0.0085, 0.205
stc.crop(tmin, tmax)
stc._data *= 1e08

rng = np.random.RandomState(42)
n_sensors, n_sources, n_times = 120, 500, stc.data.shape[1]
G = rng.randn(n_sensors, n_sources)
G /= G.std(axis=0)

X = 0.2 * rng.randn(n_sources, n_times)
X[0] += stc.data[0]
X[1] += stc.data[1]
X[2] += stc.data[2]
X[3] += stc.data[3]

M = np.dot(G, X)


def tf_solver_prox(M, G, alpha_space, alpha_time, lipschitz_constant, phi,
                   phiT, wsize=64, tstep=4, n_orient=1, maxit=200,
                   tol=1e-8, update_alpha=False, hp_iter=1, a=1, b=1):
    import time
    timeline = []  # track time

    n_sensors, n_times = M.shape
    n_sources = G.shape[1]

    n_step = np.ceil(n_times / tstep.astype(float)).astype(int)
    # n_freq = wsize / 2 + 1
    n_freq = wsize // 2 + 1
    shape = (-1, n_freq, n_step)
    n_coefs = n_step * n_freq

    for i_iter in range(hp_iter):
        Z = np.zeros((n_sources, n_coefs.sum()), dtype=np.complex)
        X = np.zeros((n_sources, n_times))
        R = M.copy()  # residual

        active_set = np.ones(n_sources, dtype=np.bool)
        Y_as = active_set.copy()
        Y_time_as = X.copy()
        Y = Z.copy()

        t = 1.0

        E = []  # track cost function

        alpha_time_lc = alpha_time / lipschitz_constant
        alpha_space_lc = alpha_space / lipschitz_constant

        for i in range(maxit):
            Z0, active_set_0 = Z, active_set  # store previous values

            if active_set.sum() < len(R) and Y_time_as is not None:
                # trick when using tight frame to do a first screen based on
                # L21 prox (L21 norms are not changed by phi)
                GTR = np.dot(G.T, R) / lipschitz_constant
                A = GTR.copy()
                A[Y_as] += Y_time_as
                _, active_set_l21 = prox_l21(A, alpha_space_lc, n_orient)
                # just compute prox_l1 on rows that won't be zeroed by prox_l21
                B = Y[active_set_l21] + phi(GTR[active_set_l21])
                if update_alpha:
                    Z, active_set_l1 = prox_l1(
                        B, alpha_time_lc[active_set_l21], n_orient)
                else:
                    Z, active_set_l1 = prox_l1(B, alpha_time_lc, n_orient)
                active_set_l21[active_set_l21] = active_set_l1
                active_set_l1 = active_set_l21
            else:
                Y += np.dot(G.T, phi(R)) / lipschitz_constant  # ISTA step
                Z, active_set_l1 = prox_l1(Y, alpha_time_lc, n_orient)

            if update_alpha:
                Z, active_set_l21 = prox_l21(Z, alpha_space_lc[active_set_l1],
                                             n_orient, shape=shape,
                                             is_stft=True)
            else:
                Z, active_set_l21 = prox_l21(Z, alpha_space_lc, n_orient,
                                             shape=shape, is_stft=True)

            active_set = active_set_l1
            active_set[active_set_l1] = active_set_l21

            if 1:  # log cost function value
                X = phiT(Z)
                RZ = M - np.dot(G[:, active_set], X)

                pobj = (0.5 * np.linalg.norm(RZ, ord='fro') ** 2 +
                        alpha_space * norm_l21_tf_multidict(Z, n_orient) +
                        alpha_time * norm_l1_tf_multidict(Z, n_orient))

                E.append(pobj)
                print("Iteration %d :: pobj %f :: n_active %d" % (i + 1,
                      pobj.min(), np.sum(active_set) / n_orient))
            else:
                print("Iteration %d" % i + 1)

            # Check convergence : max(abs(Z - Z0)) < tol
            stop = (safe_max_abs(Z, True - active_set_0[active_set]) < tol and
                    safe_max_abs(Z0, True - active_set[active_set_0]) < tol and
                    safe_max_abs_diff(Z, active_set_0[active_set],
                                      Z0, active_set[active_set_0]) < tol)

            t_start = 0.
            timeline.append(time.time() - t_start)

            if stop:
                print('Convergence reached !')
                break

            # FISTA 2 steps
            # compute efficiently : Y = Z + ((t0 - 1.0) / t) * (Z - Z0)
            t0 = t
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t ** 2))
            Y.fill(0.0)
            dt = ((t0 - 1.0) / t)
            Y[active_set] = (1.0 + dt) * Z
            if len(Z0):
                Y[active_set_0] -= dt * Z0
            Y_as = active_set_0 | active_set

            Y_time_as = phiT(Y[Y_as])
            R = M - np.dot(G[:, Y_as], Y_time_as)

        X = phiT(Z)
        if active_set.sum() == 0:
            raise Exception("No active dipoles found. "
                            "alpha_space is too big.")
        # print("alpha at iter %d: %f" % (i, alpha))

        if update_alpha:
            # l21_Z = np.sqrt(np.sum((np.abs(X) ** 2.), axis=1))
            l21_Z = np.sqrt(stft_norm2_multidict(Z).reshape(-1,
                            n_orient).sum(axis=1))
            # np.sqrt(stft_norm2_multidict(Z).reshape(-1, n_orient).sum(axis=1)
            # l1_Z = np.sum(np.abs(Z), axis=1)
            l1_Z = np.sqrt(np.sum((np.abs(Z) ** 2.).reshape((n_orient, -1),
                                                            order='F'),
                                  axis=0)).reshape(-1, 500)
            1/0
            alpha_space[active_set] = (62. + a) / (l21_Z + b)
            alpha_time[active_set] = (62. + a) / (l1_Z + b)

        if update_alpha:
            out = X, active_set, E, alpha_space, alpha_time
        else:
            out = X, active_set, E, alpha_space, alpha_time

    return out


def prox_l1(Y, alpha, n_orient):
    n_positions = Y.shape[0] // n_orient
    norms = np.sqrt(np.sum((np.abs(Y) ** 2).T.reshape(-1, n_orient), axis=1))
    if np.shape(alpha):
        alpha = np.tile(alpha, [Y.shape[1], 1]).T.reshape(-1, n_orient)[:, 0]
    # Ensure shrink is >= 0 while avoiding any division by zero
    shrink = np.maximum(1.0 - alpha / np.maximum(norms, alpha), 0.0)
    shrink = shrink.reshape(-1, n_positions).T
    active_set = np.any(shrink > 0.0, axis=1)
    shrink = shrink[active_set]
    if n_orient > 1:
        active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
    Y = Y[active_set]
    if len(Y) > 0:
        for o in range(n_orient):
            Y[o::n_orient] *= shrink
    return Y, active_set


def prox_l21(Y, alpha, n_orient, shape=None, is_stft=False):
    if len(Y) == 0:
        return np.zeros_like(Y), np.zeros((0,), dtype=np.bool)
    # if shape is not None:
    #     shape_init = Y.shape
    #     Y = Y.reshape(*shape)
    n_positions = Y.shape[0] // n_orient

    if is_stft:
        rows_norm = np.sqrt(stft_norm2_multidict(Y).reshape(n_positions,
                            -1).sum(axis=1))
    else:
        rows_norm = np.sqrt(np.sum((np.abs(Y) ** 2).reshape(n_positions, -1),
                                   axis=1))
    # Ensure shrink is >= 0 while avoiding any division by zero
    shrink = np.maximum(1.0 - alpha / np.maximum(rows_norm, alpha), 0.0)
    active_set = shrink > 0.0
    if n_orient > 1:
        active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
        shrink = np.tile(shrink[:, None], [1, n_orient]).ravel()
    Y = Y[active_set]
    # if shape is None:
    #     Y *= shrink[active_set][:, np.newaxis]
    # else:
    #     Y *= shrink[active_set][:, np.newaxis, np.newaxis]
    #     Y = Y.reshape(-1, *shape_init[1:])
    Y *= shrink[active_set][:, np.newaxis]
    return Y, active_set


def safe_max_abs(A, ia):
    """Compute np.max(np.abs(A[ia])) possible with empty A"""
    if np.sum(ia):  # ia is not empty
        return np.max(np.abs(A[ia]))
    else:
        return 0.


def safe_max_abs_diff(A, ia, B, ib):
    """Compute np.max(np.abs(A)) possible with empty A"""
    A = A[ia] if np.sum(ia) else 0.0
    B = B[ib] if np.sum(ia) else 0.0
    return np.max(np.abs(A - B))


def stft_norm2_multidict(Z):
    """Compute squared l2-norm with multiple dictionaries"""
    return (np.abs(Z) ** 2.).sum(axis=1)


def norm_l21_tf_multidict(Z, n_orient):
    if Z.shape[0]:
        l21_norm = np.sqrt(stft_norm2_multidict(Z).reshape(-1,
                           n_orient).sum(axis=1)).sum()
    else:
        l21_norm = 0.
    return l21_norm


def norm_l1_tf_multidict(Z, n_orient):
    if Z.shape[0]:
        l1_norm = np.sqrt(np.sum((np.abs(Z) ** 2.).reshape((n_orient, -1),
                                                           order='F'),
                                 axis=0)).sum()
    else:
        l1_norm = 0.
    return l1_norm


def alpha_max_fun(alpha, rho, GTM):
    thresh = np.sign(GTM) * np.maximum(np.abs(GTM) - alpha * rho, 0.0)
    return (stft_norm2_multidict(
        thresh[None, :]).sum() - ((1 - rho) * alpha) ** 2)


def compute_alpha_max(G, M, phi, rho, n_orient):
    if rho:
        from scipy.optimize import brentq
        n_positions = G.shape[1] // n_orient
        alpha_max = 0.0
        for idx in xrange(n_positions):
            idx_start = idx * n_orient
            idx_end = idx_start + n_orient
            GTM = np.abs(phi(np.dot(G[:, idx_start:idx_end].T, M))) ** 2.
            GTM = np.sqrt(np.sum(GTM, axis=0))
            max_alpha = np.max(np.abs(GTM)) / rho
            alpha_max_tmp = brentq(alpha_max_fun, 0., max_alpha,
                                   args=(rho, GTM))
            if alpha_max_tmp > alpha_max:
                alpha_max = alpha_max_tmp
    else:
        n_positions = G.shape[1] // n_orient
        alpha_max = stft_norm2_multidict(phi(np.dot(G.T, M)))
        alpha_max = np.sqrt(alpha_max.reshape(n_positions, -1).sum(axis=1))
        alpha_max = np.max(alpha_max)

    return alpha_max


class _Phi(object):
    """Util class to have phi stft as callable without using
    a lambda that does not pickle"""
    def __init__(self, wsize, tstep, n_coefs):
        self.wsize = wsize
        self.tstep = tstep
        self.n_coefs = n_coefs

    def __call__(self, x):
        return np.hstack(
            [stft(x, self.wsize[i], self.tstep[i],
                  verbose=False).reshape(-1, self.n_coefs[i])
             for i in range(len(self.n_coefs))]) / np.sqrt(len(self.n_coefs))


class _PhiT(object):
    """Util class to have phi.T istft as callable without using
    a lambda that does not pickle"""
    def __init__(self, tstep, n_freq, n_step, n_times):
        self.tstep = tstep
        self.n_freq = n_freq
        self.n_step = n_step
        self.n_times = n_times

    def __call__(self, z):
        x_out = np.zeros((z.shape[0], self.n_times))
        n_coefs = self.n_freq * self.n_step
        if len(n_coefs) > 1:
            z_ = np.array_split(z, np.cumsum(n_coefs)[:-1], axis=1)
        else:
            z_ = [z]
        for i in range(len(z_)):
            x_out += istft(z_[i].reshape(-1, self.n_freq[i], self.n_step[i]),
                           self.tstep[i], self.n_times)
        return x_out / np.sqrt(len(n_coefs))


def tf_lipschitz_constant(M, G, phi, phiT, tol=1e-3, verbose=None):
    """Compute lipschitz constant for FISTA

    It uses a power iteration method.
    """
    n_times = M.shape[1]
    n_points = G.shape[1]
    iv = np.ones((n_points, n_times), dtype=np.float)
    v = phi(iv)
    L = 1e100
    for it in range(100):
        L_old = L
        print('Lipschitz estimation: iteration = %d' % it)
        iv = np.real(phiT(v))
        Gv = np.dot(G, iv)
        GtGv = np.dot(G.T, Gv)
        w = phi(GtGv)
        L = np.max(np.abs(w))  # l_inf norm
        v = w / L
        if abs((L - L_old) / L_old) < tol:
            break
    return L


n_orient = 1
wsize, tstep = np.array([16, 64]), np.array([2, 4])
n_sensors, n_times = M.shape

n_step = np.ceil(n_times / tstep.astype(float)).astype(int)
n_freq = wsize // 2 + 1
n_coefs = (n_step * n_freq).astype(int)
phi = _Phi(wsize, tstep, n_coefs)
phiT = _PhiT(tstep, n_freq, n_step, n_times)

alpha_max = compute_alpha_max(G, M, phi, rho=0., n_orient=1)  # XXX rho :/
alpha_max *= 0.01
G /= alpha_max

# lc = 1.05 * np.linalg.norm(G, ord=2) ** 2
lc = tf_lipschitz_constant(M, G, phi, phiT)

mode = alpha_max * 100 / 2.
b = 1.
a = mode * b + 1.

# Alphas
alpha_space = 20.
alpha_time = .35

alpha_space = 1. * np.ones((G.shape[1],))
alpha_time = .1 * np.ones((G.shape[1],))
update_alpha = True
hp_iter = 5
out = tf_solver_prox(M, G, alpha_space, alpha_time, lc, phi, phiT, wsize=wsize,
                     tstep=tstep, maxit=50, tol=1e-6, n_orient=1,
                     hp_iter=hp_iter, update_alpha=update_alpha, a=a, b=b)

X_est, active_set, E, alpha_space, alpha_time = out
X_est /= alpha_max

plt.close('all')
# plt.plot(np.log10(E - np.min(E)))
# plt.plot(E)
plt.plot(X_est.T)
plt.show()
