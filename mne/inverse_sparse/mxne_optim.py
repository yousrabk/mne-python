from __future__ import print_function
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#
# License: Simplified BSD

from copy import deepcopy
import warnings
from math import sqrt, ceil
import numpy as np
from scipy import linalg

from .mxne_debiasing import compute_bias
from ..utils import logger, verbose, sum_squared
from ..time_frequency.stft import stft_norm2, stft, istft
from ..externals.six.moves import xrange as range


def groups_norm2(A, n_orient):
    """compute squared L2 norms of groups inplace"""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l2inf(A, n_orient, copy=True):
    """L2-inf norm"""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return sqrt(np.max(groups_norm2(A, n_orient)))


def norm_l21(A, n_orient, copy=True):
    """L21 norm"""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sum(np.sqrt(groups_norm2(A, n_orient)))


def prox_l21(Y, alpha, n_orient, shape=None, is_stft=False):
    """proximity operator for l21 norm

    L2 over columns and L1 over rows => groups contain n_orient rows.

    It can eventually take into account the negative frequencies
    when a complex value is passed and is_stft=True.

    Example
    -------
    >>> Y = np.tile(np.array([0, 4, 3, 0, 0], dtype=np.float), (2, 1))
    >>> Y = np.r_[Y, np.zeros_like(Y)]
    >>> print(Y)
    [[ 0.  4.  3.  0.  0.]
     [ 0.  4.  3.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    >>> Yp, active_set = prox_l21(Y, 2, 2)
    >>> print(Yp)
    [[ 0.          2.86862915  2.15147186  0.          0.        ]
     [ 0.          2.86862915  2.15147186  0.          0.        ]]
    >>> print(active_set)
    [ True  True False False]
    """
    if len(Y) == 0:
        return np.zeros_like(Y), np.zeros((0,), dtype=np.bool)
    if shape is not None:
        shape_init = Y.shape
        Y = Y.reshape(*shape)
    n_positions = Y.shape[0] // n_orient

    if is_stft:
        rows_norm = np.sqrt(stft_norm2(Y).reshape(n_positions, -1).sum(axis=1))
    else:
        rows_norm = np.sqrt((Y * Y.conj()).real.reshape(n_positions,
                                                        -1).sum(axis=1))
    # Ensure shrink is >= 0 while avoiding any division by zero
    shrink = np.maximum(1.0 - alpha / np.maximum(rows_norm, alpha), 0.0)
    active_set = shrink > 0.0
    if n_orient > 1:
        active_set = np.tile(active_set[:, None], [1, n_orient]).ravel()
        shrink = np.tile(shrink[:, None], [1, n_orient]).ravel()
    Y = Y[active_set]
    if shape is None:
        Y *= shrink[active_set][:, np.newaxis]
    else:
        Y *= shrink[active_set][:, np.newaxis, np.newaxis]
        Y = Y.reshape(-1, *shape_init[1:])
    return Y, active_set


def prox_l1(Y, alpha, n_orient):
    """proximity operator for l1 norm with multiple orientation support

    L2 over orientation and L1 over position (space + time)

    Example
    -------
    >>> Y = np.tile(np.array([1, 2, 3, 2, 0], dtype=np.float), (2, 1))
    >>> Y = np.r_[Y, np.zeros_like(Y)]
    >>> print(Y)
    [[ 1.  2.  3.  2.  0.]
     [ 1.  2.  3.  2.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    >>> Yp, active_set = prox_l1(Y, 2, 2)
    >>> print(Yp)
    [[ 0.          0.58578644  1.58578644  0.58578644  0.        ]
     [ 0.          0.58578644  1.58578644  0.58578644  0.        ]]
    >>> print(active_set)
    [ True  True False False]
    """
    n_positions = Y.shape[0] // n_orient
    norms = np.sqrt((Y * Y.conj()).real.T.reshape(-1, n_orient).sum(axis=1))
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


def dgap_l21(M, G, X, active_set, alpha, n_orient):
    """Duality gaps for the mixed norm inverse problem

    For details see:
    Gramfort A., Kowalski M. and Hamalainen, M,
    Mixed-norm estimates for the M/EEG inverse problem using accelerated
    gradient methods, Physics in Medicine and Biology, 2012
    http://dx.doi.org/10.1088/0031-9155/57/7/1937

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_active)
        The gain matrix a.k.a. lead field.
    X : array, shape (n_active, n_times)
        Sources
    active_set : array of bool
        Mask of active sources
    alpha : float
        Regularization parameter
    n_orient : int
        Number of dipoles per locations (typically 1 or 3)

    Returns
    -------
    gap : float
        Dual gap
    pobj : float
        Primal cost
    dobj : float
        Dual cost. gap = pobj - dobj
    R : array, shape (n_sensors, n_times)
        Current residual of M - G * X
    """
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    penalty = norm_l21(X, n_orient, copy=True)
    nR2 = sum_squared(R)
    pobj = 0.5 * nR2 + alpha * penalty
    dual_norm = norm_l2inf(np.dot(G.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    dobj = 0.5 * (scaling ** 2) * nR2 + scaling * np.sum(R * GX)
    gap = pobj - dobj
    return gap, pobj, dobj, R


@verbose
def _mixed_norm_solver_prox(M, G, alpha, lipschitz_constant, maxit=200,
                            tol=1e-8, verbose=None, init=None, n_orient=1):
    """Solves L21 inverse problem with proximal iterations and FISTA"""
    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape

    if n_sources < n_sensors:
        gram = np.dot(G.T, G)
        GTM = np.dot(G.T, M)
    else:
        gram = None

    if init is None:
        X = 0.0
        R = M.copy()
        if gram is not None:
            R = np.dot(G.T, R)
    else:
        X = init
        if gram is None:
            R = M - np.dot(G, X)
        else:
            R = GTM - np.dot(gram, X)

    t = 1.0
    Y = np.zeros((n_sources, n_times))  # FISTA aux variable
    E = []  # track cost function

    active_set = np.ones(n_sources, dtype=np.bool)  # start with full AS

    for i in range(maxit):
        X0, active_set_0 = X, active_set  # store previous values
        if gram is None:
            Y += np.dot(G.T, R) / lipschitz_constant  # ISTA step
        else:
            Y += R / lipschitz_constant  # ISTA step
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

        gap, pobj, dobj, _ = dgap_l21(M, G, X, active_set, alpha, n_orient)
        E.append(pobj)
        logger.debug("pobj : %s -- gap : %s" % (pobj, gap))
        if gap < tol:
            logger.debug('Convergence reached ! (gap: %s < %s)' % (gap, tol))
            break
    return X, active_set, E


@verbose
def _mixed_norm_solver_cd(M, G, alpha, lipschitz_constant, maxit=10000,
                          tol=1e-8, verbose=None, init=None, n_orient=1):
    """Solves L21 inverse problem with coordinate descent"""
    from sklearn.linear_model.coordinate_descent import MultiTaskLasso

    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape

    if init is not None:
        init = init.T

    clf = MultiTaskLasso(alpha=alpha / len(M), tol=tol, normalize=False,
                         fit_intercept=False, max_iter=maxit,
                         warm_start=True)
    clf.coef_ = init
    clf.fit(G, M)

    X = clf.coef_.T
    active_set = np.any(X, axis=1)
    X = X[active_set]
    gap, pobj, dobj, _ = dgap_l21(M, G, X, active_set, alpha, n_orient)
    return X, active_set, pobj


@verbose
def _mixed_norm_solver_bcd(M, G, alpha, lipschitz_constant, maxit=200,
                           tol=1e-8, verbose=None, init=None, n_orient=1):
    """Solves L21 inverse problem with block coordinate descent"""
    # First make G fortran for faster access to blocks of columns
    G = np.asfortranarray(G)

    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape
    n_positions = n_sources // n_orient

    if init is None:
        X = np.zeros((n_sources, n_times))
        R = M.copy()
    else:
        X = init
        R = M - np.dot(G, X)

    E = []  # track cost function

    active_set = np.zeros(n_sources, dtype=np.bool)  # start with full AS

    alpha_lc = alpha / lipschitz_constant

    for i in range(maxit):
        for j in range(n_positions):
            idx = slice(j * n_orient, (j + 1) * n_orient)

            G_j = G[:, idx]
            X_j = X[idx]

            X_j_new = np.dot(G_j.T, R) / lipschitz_constant[j]

            was_non_zero = np.any(X_j)
            if was_non_zero:
                R += np.dot(G_j, X_j)
                X_j_new += X_j

            block_norm = linalg.norm(X_j_new, 'fro')
            if block_norm <= alpha_lc[j]:
                X_j.fill(0.)
                active_set[idx] = False
            else:
                shrink = np.maximum(1.0 - alpha_lc[j] / block_norm, 0.0)
                X_j_new *= shrink
                R -= np.dot(G_j, X_j_new)
                X_j[:] = X_j_new
                active_set[idx] = True

        gap, pobj, dobj, _ = dgap_l21(M, G, X[active_set], active_set, alpha,
                                      n_orient)
        E.append(pobj)
        logger.debug("Iteration %d :: pobj %f :: dgap %f :: n_active %d" % (
                     i + 1, pobj, gap, np.sum(active_set) / n_orient))

        if gap < tol:
            logger.debug('Convergence reached ! (gap: %s < %s)' % (gap, tol))
            break

    X = X[active_set]

    return X, active_set, E


@verbose
def mixed_norm_solver(M, G, alpha, maxit=3000, tol=1e-8, verbose=None,
                      active_set_size=50, debias=True, n_orient=1,
                      solver='auto'):
    """Solves L1/L2 mixed-norm inverse problem with active set strategy

    Algorithm is detailed in:
    Gramfort A., Kowalski M. and Hamalainen, M,
    Mixed-norm estimates for the M/EEG inverse problem using accelerated
    gradient methods, Physics in Medicine and Biology, 2012
    http://dx.doi.org/10.1088/0031-9155/57/7/1937

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_dipoles)
        The gain matrix a.k.a. lead field.
    alpha : float
        The regularization parameter. It should be between 0 and 100.
        A value of 100 will lead to an empty active set (no active source).
    maxit : int
        The number of iterations.
    tol : float
        Tolerance on dual gap for convergence checking.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    active_set_size : int
        Size of active set increase at each iteration.
    debias : bool
        Debias source estimates.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    solver : 'prox' | 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization.

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function over the iterations.
    """
    n_dipoles = G.shape[1]
    n_positions = n_dipoles // n_orient
    n_sensors, n_times = M.shape
    alpha_max = norm_l2inf(np.dot(G.T, M), n_orient, copy=False)
    logger.info("-- ALPHA MAX : %s" % alpha_max)
    alpha = float(alpha)

    has_sklearn = True
    try:
        from sklearn.linear_model.coordinate_descent import MultiTaskLasso  # noqa
    except ImportError:
        has_sklearn = False

    if solver == 'auto':
        if has_sklearn and (n_orient == 1):
            solver = 'cd'
        else:
            solver = 'bcd'

    if solver == 'cd':
        if n_orient == 1 and not has_sklearn:
            warnings.warn("Scikit-learn >= 0.12 cannot be found. "
                          "Using block coordinate descent instead of "
                          "coordinate descent.")
            solver = 'bcd'
        if n_orient > 1:
            warnings.warn("Coordinate descent is only available for fixed "
                          "orientation. Using block coordinate descent "
                          "instead of coordinate descent")
            solver = 'bcd'

    if solver == 'cd':
        logger.info("Using coordinate descent")
        l21_solver = _mixed_norm_solver_cd
        lc = None
    elif solver == 'bcd':
        logger.info("Using block coordinate descent")
        l21_solver = _mixed_norm_solver_bcd
        G = np.asfortranarray(G)
        if n_orient == 1:
            lc = np.sum(G * G, axis=0)
        else:
            lc = np.empty(n_positions)
            for j in range(n_positions):
                G_tmp = G[:, (j * n_orient):((j + 1) * n_orient)]
                lc[j] = linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)
    else:
        logger.info("Using proximal iterations")
        l21_solver = _mixed_norm_solver_prox
        lc = 1.01 * linalg.norm(G, ord=2) ** 2

    if active_set_size is not None:
        E = list()
        X_init = None
        active_set = np.zeros(n_dipoles, dtype=np.bool)
        idx_large_corr = np.argsort(groups_norm2(np.dot(G.T, M), n_orient))
        new_active_idx = idx_large_corr[-active_set_size:]
        if n_orient > 1:
            new_active_idx = (n_orient * new_active_idx[:, None] +
                              np.arange(n_orient)[None, :]).ravel()
        active_set[new_active_idx] = True
        as_size = np.sum(active_set)
        for k in range(maxit):
            if solver == 'bcd':
                lc_tmp = lc[active_set[::n_orient]]
            elif solver == 'cd':
                lc_tmp = None
            else:
                lc_tmp = 1.01 * linalg.norm(G[:, active_set], ord=2) ** 2
            X, as_, _ = l21_solver(M, G[:, active_set], alpha, lc_tmp,
                                   maxit=maxit, tol=tol, init=X_init,
                                   n_orient=n_orient)
            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            gap, pobj, dobj, R = dgap_l21(M, G, X, active_set, alpha,
                                          n_orient)
            E.append(pobj)
            logger.info("Iteration %d :: pobj %f :: dgap %f ::"
                        "n_active_start %d :: n_active_end %d" % (
                            k + 1, pobj, gap, as_size // n_orient,
                            np.sum(active_set) // n_orient))
            if gap < tol:
                logger.info('Convergence reached ! (gap: %s < %s)'
                            % (gap, tol))
                break

            # add sources if not last iteration
            if k < (maxit - 1):
                idx_large_corr = np.argsort(groups_norm2(np.dot(G.T, R),
                                            n_orient))
                new_active_idx = idx_large_corr[-active_set_size:]
                if n_orient > 1:
                    new_active_idx = (n_orient * new_active_idx[:, None] +
                                      np.arange(n_orient)[None, :])
                    new_active_idx = new_active_idx.ravel()
                active_set[new_active_idx] = True
                idx_active_set = np.where(active_set)[0]
                as_size = np.sum(active_set)
                X_init = np.zeros((as_size, n_times), dtype=X.dtype)
                idx = np.searchsorted(idx_active_set, idx_old_active_set)
                X_init[idx] = X
        else:
            logger.warning('Did NOT converge ! (gap: %s > %s)' % (gap, tol))
    else:
        X, active_set, E = l21_solver(M, G, alpha, lc, maxit=maxit,
                                      tol=tol, n_orient=n_orient, init=None)

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    logger.info('Final active set size: %s' % (np.sum(active_set) // n_orient))

    return X, active_set, E


@verbose
def iterative_mixed_norm_solver(M, G, alpha, n_mxne_iter, maxit=3000,
                                tol=1e-8, verbose=None, active_set_size=50,
                                debias=True, n_orient=1, solver='auto'):
    """Solves L0.5/L2 mixed-norm inverse problem with active set strategy

    Algorithm is detailed in:

    Strohmeier D., Haueisen J., and Gramfort A.:
    Improved MEG/EEG source localization with reweighted mixed-norms,
    4th International Workshop on Pattern Recognition in Neuroimaging,
    Tuebingen, 2014

    Parameters
    ----------
    M : array, shape (n_sensors, n_times)
        The data.
    G : array, shape (n_sensors, n_dipoles)
        The gain matrix a.k.a. lead field.
    alpha : float
        The regularization parameter. It should be between 0 and 100.
        A value of 100 will lead to an empty active set (no active source).
    n_mxne_iter : int
        The number of MxNE iterations. If > 1, iterative reweighting
        is applied.
    maxit : int
        The number of iterations.
    tol : float
        Tolerance on dual gap for convergence checking.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    active_set_size : int
        Size of active set increase at each iteration.
    debias : bool
        Debias source estimates.
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    solver : 'prox' | 'cd' | 'bcd' | 'auto'
        The algorithm to use for the optimization.

    Returns
    -------
    X : array, shape (n_active, n_times)
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function over the iterations.
    """
    def g(w):
        return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    def gprime(w):
        return 2. * np.repeat(g(w), n_orient).ravel()

    E = list()

    active_set = np.ones(G.shape[1], dtype=np.bool)
    weights = np.ones(G.shape[1])
    X = np.zeros((G.shape[1], M.shape[1]))

    for k in range(n_mxne_iter):
        X0 = X.copy()
        active_set_0 = active_set.copy()
        G_tmp = G[:, active_set] * weights[np.newaxis, :]

        if active_set_size is not None:
            if np.sum(active_set) > (active_set_size * n_orient):
                X, _active_set, _ = mixed_norm_solver(
                    M, G_tmp, alpha, debias=False, n_orient=n_orient,
                    maxit=maxit, tol=tol, active_set_size=active_set_size,
                    solver=solver, verbose=verbose)
            else:
                X, _active_set, _ = mixed_norm_solver(
                    M, G_tmp, alpha, debias=False, n_orient=n_orient,
                    maxit=maxit, tol=tol, active_set_size=None, solver=solver,
                    verbose=verbose)
        else:
            X, _active_set, _ = mixed_norm_solver(
                M, G_tmp, alpha, debias=False, n_orient=n_orient,
                maxit=maxit, tol=tol, active_set_size=None, solver=solver,
                verbose=verbose)

        logger.info('active set size %d' % (_active_set.sum() / n_orient))

        if _active_set.sum() > 0:
            active_set[active_set] = _active_set

            # Reapply weights to have correct unit
            X *= weights[_active_set][:, np.newaxis]
            weights = gprime(X)
            p_obj = 0.5 * linalg.norm(M - np.dot(G[:, active_set],  X),
                                      'fro') ** 2. + alpha * np.sum(g(X))
            E.append(p_obj)

            # Check convergence
            if ((k >= 1) and np.all(active_set == active_set_0) and
                    np.all(np.abs(X - X0) < tol)):
                print('Convergence reached after %d reweightings!' % k)
                break
        else:
            active_set = np.zeros_like(active_set)
            p_obj = 0.5 * linalg.norm(M) ** 2.
            E.append(p_obj)
            break

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    return X, active_set, E


###############################################################################
# TF-MxNE
class _Phi(object):
    """Util class to have phi stft as callable without using
    a lambda that does not pickle"""
    def __init__(self, wsize, tstep, n_coefs):
        self.wsize = wsize
        self.tstep = tstep
        self.n_coefs = n_coefs

    def __call__(self, x):
        return np.hstack([stft(x, self.wsize[i], self.tstep[i],
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
                  order='F'), axis=0)).sum()
    else:
        l1_norm = 0.
    return l1_norm


def test_multidict(x, wsize=np.array([64]), tstep=np.array([4])):
    n_sensors, n_times = x.shape

    n_step = np.ceil(n_times / tstep.astype(float)).astype(int)
    # n_freq = wsize / 2 + 1
    n_freq = wsize
    n_coefs = (n_step * n_freq).astype(int)
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)

    z_out = phi(x)
    x_out = phiT(z_out)

    E1 = (x ** 2).sum(axis=1)
    E2 = (x_out ** 2).sum(axis=1)
    E3 = stft_norm2_multidict(z_out)

    return z_out, x_out, E1, E2, E3


def prox_l21(Y, alpha, n_orient, shape=None, is_stft=False):
    """proximity operator for l21 norm

    L2 over columns and L1 over rows => groups contain n_orient rows.

    It can eventually take into account the negative frequencies
    when a complex value is passed and is_stft=True.

    Example
    -------
    >>> Y = np.tile(np.array([0, 4, 3, 0, 0], dtype=np.float), (2, 1))
    >>> Y = np.r_[Y, np.zeros_like(Y)]
    >>> print(Y)
    [[ 0.  4.  3.  0.  0.]
     [ 0.  4.  3.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    >>> Yp, active_set = prox_l21(Y, 2, 2)
    >>> print(Yp)
    [[ 0.          2.86862915  2.15147186  0.          0.        ]
     [ 0.          2.86862915  2.15147186  0.          0.        ]]
    >>> print(active_set)
    [ True  True False False]
    """
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


def prox_l1(Y, alpha, n_orient):
    """proximity operator for l1 norm with multiple orientation support

    L2 over orientation and L1 over position (space + time)

    Example
    -------
    >>> Y = np.tile(np.array([1, 2, 3, 2, 0], dtype=np.float), (2, 1))
    >>> Y = np.r_[Y, np.zeros_like(Y)]
    >>> print(Y)
    [[ 1.  2.  3.  2.  0.]
     [ 1.  2.  3.  2.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]]
    >>> Yp, active_set = prox_l1(Y, 2, 2)
    >>> print(Yp)
    [[ 0.          0.58578644  1.58578644  0.58578644  0.        ]
     [ 0.          0.58578644  1.58578644  0.58578644  0.        ]]
    >>> print(active_set)
    [ True  True False False]
    """
    n_positions = Y.shape[0] // n_orient
    norms = np.sqrt(np.sum((np.abs(Y) ** 2).T.reshape(-1, n_orient), axis=1))
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


@verbose
def _tf_mixed_norm_solver_bcd_(M, G, Z, active_set, alpha_space, alpha_time,
                               lipschitz_constant, phi, phiT, w_space=None,
                               w_time=None, wsize=64, tstep=4, n_orient=1,
                               maxit=200, tol=1e-8, check_convergence=True,
                               log_objective=True, verbose=None):
    """Solves TF L21+L1 inverse solver

    Algorithm is detailed in:

    A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
    Time-Frequency Mixed-Norm Estimates: Sparse M/EEG imaging with
    non-stationary source activations
    Neuroimage, Volume 70, 15 April 2013, Pages 410-422, ISSN 1053-8119,
    DOI: 10.1016/j.neuroimage.2012.12.051.

    Functional Brain Imaging with M/EEG Using Structured Sparsity in
    Time-Frequency Dictionaries
    Gramfort A., Strohmeier D., Haueisen J., Hamalainen M. and Kowalski M.
    INFORMATION PROCESSING IN MEDICAL IMAGING
    Lecture Notes in Computer Science, 2011, Volume 6801/2011,
    600-611, DOI: 10.1007/978-3-642-22092-0_49
    http://dx.doi.org/10.1007/978-3-642-22092-0_49

    Parameters
    ----------
    M : array
        The data.
    G : array
        The forward operator.
    alpha_space : float
        The spatial regularization parameter. It should be between 0 and 100.
    alpha_time : float
        The temporal regularization parameter. The higher it is the smoother
        will be the estimated time series.
    wsize: int
        length of the STFT window in samples (must be a multiple of 4).
    tstep: int
        step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    maxit : int
        The number of iterations.
    tol : float
        If absolute difference between estimates at 2 successive iterations
        is lower than tol, the convergence is reached.
    log_objective : bool
        If True, the value of the minimized objective function is computed
        and stored at every iteration.
    lipschitz_constant : float | None
        The lipschitz constant of the spatio temporal linear operator.
        If None it is estimated.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    X : array
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function at each iteration. If log_objective
        is False, it will be empty.
    """
    # First make G fortran for faster access to blocks of columns
    G = np.asfortranarray(G)

    n_sensors, n_times = M.shape
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    G = dict(zip(np.arange(n_positions), np.hsplit(G, n_positions)))
    R = M.copy()  # residual
    active = np.where(active_set == True)[0][::n_orient] // n_orient
    for idx in active:
        R -= np.dot(G[idx], phiT(Z[idx]))

    E = []  # track cost function

    if w_time is None:
        alpha_time_lc = alpha_time / lipschitz_constant
    else:
        alpha_time_lc = alpha_time * w_time / lipschitz_constant[:, None]
    if w_space is None:
        alpha_space_lc = alpha_space / lipschitz_constant
    else:
        alpha_space_lc = alpha_space * w_space / lipschitz_constant

    for i in xrange(maxit):
        val_norm_l21_tf = 0.0
        val_norm_l1_tf = 0.0
        max_diff = 0.0
        for j in xrange(n_positions):
            ids = j * n_orient
            ide = ids + n_orient

            G_j = G[j]
            Z_j = Z[j]
            active_set_j = active_set[ids:ide]
            Z0 = deepcopy(Z_j)

            was_active = np.any(active_set_j)

            # gradient step
            PhiGTR = phi(np.dot(G_j.T, R)) / lipschitz_constant[j]
            Z_j_new = PhiGTR.copy()
            if was_active:
                Z_j_new += Z_j
                R += np.dot(G_j, phiT(Z_j))

            col_norm = np.sqrt(np.sum(np.abs(Z_j_new) ** 2, axis=0))

            if np.all(col_norm <= alpha_time_lc[j]):
                Z[j] = 0.0
                active_set_j[:] = False
            else:
                # l1
                shrink = np.maximum(1.0 - alpha_time_lc[j] / col_norm, 0.0)
                Z_j_new *= shrink[np.newaxis, :]

                # l21
                row_norm = np.sqrt(stft_norm2_multidict(Z_j_new).sum())
                if row_norm <= alpha_space_lc[j]:
                    Z[j] = 0.0
                    active_set_j[:] = False
                else:
                    shrink = np.maximum(1.0 - alpha_space_lc[j] / row_norm, 0.0)
                    Z[j] = Z_j_new.copy() * shrink
                    active_set_j[:] = True
                    R -= np.dot(G_j, phiT(Z[j]))
                    if w_space is None:
                        val_norm_l21_tf += norm_l21_tf_multidict(Z[j],
                            n_orient)
                    else:
                        val_norm_l21_tf += w_space[j] * norm_l21_tf_multidict(
                            Z[j], n_orient)
                    if w_time is None:
                        val_norm_l1_tf += norm_l1_tf_multidict(Z[j],
                            n_orient)
                    else:
                        val_norm_l1_tf += norm_l1_tf_multidict(
                            Z[j] * w_time[j][None, :], n_orient)

                max_diff = np.maximum(max_diff, np.amax(np.abs(Z[j] - Z0)))

        if log_objective:  # log cost function value
            pobj = (0.5 * (R ** 2.).sum() + alpha_space * val_norm_l21_tf +
                    alpha_time * val_norm_l1_tf)
            E.append(pobj)
        else:
            logger.info("Iteration %d" % i + 1)

        if check_convergence:
            if i > 0:
                max_diff = np.max(np.abs(E[-2] - E[-1]) / E[-2])
                if (max_diff < tol):
                    print('BCD: Convergence reached after %d iterations!' % i)
                    break
                # if max_diff < tol:
                #     logger.info("Convergence reached !")
                #     break

    return Z, active_set, E


@verbose
def _tf_mixed_norm_solver_bcd_active_set(
    M, G, alpha_space, alpha_time, lipschitz_constant, phi, phiT, Z_init=None,
    w_space=None, w_time=None, wsize=64, tstep=4, n_orient=1, maxit=200,
    tol=1e-8, log_objective=True, t_start=0., verbose=None):
    """Solves L21 inverse problem with block coordinate descent"""
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    if Z_init is None:
        Z = dict.fromkeys(np.arange(n_positions), 0.0)
        active_set = np.zeros(n_sources, dtype=np.bool)
    else:
        active_set = np.zeros(n_sources, dtype=np.bool)
        active = list()
        for i in xrange(n_positions):
            if np.any(Z_init[i * n_orient:(i + 1) * n_orient]):
                active_set[i * n_orient:(i + 1) * n_orient] = True
                active.append(i)
        Z = dict.fromkeys(np.arange(n_positions), 0.0)
        if len(active):
            Z.update(dict(zip(active, np.vsplit(Z_init[active_set],
                     len(active)))))

    Z, active_set, E = _tf_mixed_norm_solver_bcd_(
        M, G, Z, active_set, alpha_space, alpha_time, lipschitz_constant,
        phi, phiT, w_space=w_space, w_time=w_time, wsize=wsize,
        tstep=tstep, n_orient=n_orient, maxit=1, tol=tol,
        log_objective=log_objective, check_convergence=False,
        verbose=verbose)

    i = 0
    while active_set.sum():
        i += 1
        active = np.where(active_set == True)[0][::n_orient] // n_orient
        Z_init = dict(zip(range(len(active)), [Z[idx] for idx in active]))
        if w_space is None:
            _w_space = None
        else:
            _w_space = w_space[active_set[::n_orient]]
        if w_time is None:
            _w_time = None
        else:
            _w_time = w_time[active_set[::n_orient]]
        Z, as_, E_tmp = _tf_mixed_norm_solver_bcd_(M,
            G[:, active_set], Z_init, np.ones(len(active) * n_orient,
            dtype=np.bool), alpha_space, alpha_time,
            lipschitz_constant[active_set[::n_orient]],
            phi, phiT, w_space=_w_space, w_time=_w_time, wsize=wsize,
            tstep=tstep, n_orient=n_orient,
            maxit=maxit, tol=tol, log_objective=log_objective,
            verbose=verbose)
        E += E_tmp

        active = np.where(active_set == True)[0][::n_orient] // n_orient
        Z_init = dict.fromkeys(np.arange(n_positions), 0.0)
        Z_init.update(dict(zip(active, list(Z.values()))))
        active_set[active_set] = as_.copy()
        active_set_0 = active_set.copy()
        Z, active_set, E_tmp = _tf_mixed_norm_solver_bcd_(M, G,
            Z_init, active_set, alpha_space, alpha_time, lipschitz_constant,
            phi, phiT, w_space=w_space, w_time=w_time, wsize=wsize,
            tstep=tstep, n_orient=n_orient, maxit=1, tol=tol,
            log_objective=log_objective, check_convergence=False,
            verbose=verbose)
        E += E_tmp
        if np.array_equal(active_set_0, active_set):
            # logger.info("Convergence reached !")
            print('active_set: Convergence reached after %d iterations!' % i)
            break

    if active_set.sum():
        Z = np.vstack([Z_ for Z_ in list(Z.values()) if np.any(Z_)])
        X = phiT(Z)
    else:
        n_sensors, n_times = M.shape
        n_step = np.ceil(n_times / tstep.astype(float)).astype(int)
        n_freq = wsize // 2 + 1
        Z = np.zeros((0, (n_step * n_freq).sum()), dtype=np.complex)
        X = np.zeros((0, n_times))


    return X, Z, active_set, E


def alpha_max_fun(alpha, rho, GTM):
    thresh = np.sign(GTM) * np.maximum(np.abs(GTM) - alpha * rho, 0.0)
    return (stft_norm2_multidict(
        thresh[None, :]).sum() - ((1 - rho) * alpha) ** 2)


def compute_alpha_max(G, M, phi, alpha_space, alpha_time, n_orient):
    n_positions = G.shape[1] // n_orient
    if alpha_time:
        from scipy.optimize import brentq
        rho = alpha_time / (alpha_space + alpha_time)
        alpha_max = 0.0
        for idx in xrange(n_positions):
            idx_k = slice(n_orient * idx, n_orient * (idx + 1))
            GTM = np.abs(np.dot(G[:, idx_k].T, M)) ** 2.
            GTM = np.sqrt(np.sum(GTM, axis=0))
            max_alpha = np.max(np.abs(GTM)) / rho
            alpha_max_tmp = brentq(alpha_max_fun, 0., max_alpha,
                                   args=(rho , GTM))
            if alpha_max_tmp > alpha_max:
                alpha_max = alpha_max_tmp
    else:
        alpha_max = stft_norm2_multidict(phi(np.dot(G.T, M)))
        alpha_max = np.sqrt(alpha_max.reshape(n_positions, -1).sum(axis=1))
        alpha_max = np.max(alpha_max)

    return alpha_max


@verbose
def tf_mixed_norm_solver(M, G, alpha_space, alpha_time, w_space=None, w_time=None,
                         wsize=64, tstep=4, n_orient=1, maxit=200, tol=1e-8,
                         log_objective=True, lipschitz_constant=None,
                         debias=True, verbose=None):
    """Solves TF L21+L1 inverse solver

    Algorithm is detailed in:

    A. Gramfort, D. Strohmeier, J. Haueisen, M. Hamalainen, M. Kowalski
    Time-Frequency Mixed-Norm Estimates: Sparse M/EEG imaging with
    non-stationary source activations
    Neuroimage, Volume 70, 15 April 2013, Pages 410-422, ISSN 1053-8119,
    DOI: 10.1016/j.neuroimage.2012.12.051.

    Functional Brain Imaging with M/EEG Using Structured Sparsity in
    Time-Frequency Dictionaries
    Gramfort A., Strohmeier D., Haueisen J., Hamalainen M. and Kowalski M.
    INFORMATION PROCESSING IN MEDICAL IMAGING
    Lecture Notes in Computer Science, 2011, Volume 6801/2011,
    600-611, DOI: 10.1007/978-3-642-22092-0_49
    http://dx.doi.org/10.1007/978-3-642-22092-0_49

    Parameters
    ----------
    M : array
        The data.
    G : array
        The forward operator.
    alpha_space : float
        The spatial regularization parameter. It should be between 0 and 100.
    alpha_time : float
        The temporal regularization parameter. The higher it is the smoother
        will be the estimated time series.
    wsize: int
        length of the STFT window in samples (must be a multiple of 4).
    tstep: int
        step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
    n_orient : int
        The number of orientation (1 : fixed or 3 : free or loose).
    maxit : int
        The number of iterations.
    tol : float
        If absolute difference between estimates at 2 successive iterations
        is lower than tol, the convergence is reached.
    log_objective : bool
        If True, the value of the minimized objective function is computed
        and stored at every iteration.
    lipschitz_constant : float | None
        The lipschitz constant of the spatio temporal linear operator.
        If None it is estimated.
    debias : bool
        Debias source estimates.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    X : array
        The source estimates.
    active_set : array
        The mask of active sources.
    E : list
        The value of the objective function at each iteration. If log_objective
        is False, it will be empty.
    """

    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape
    n_positions = n_sources // n_orient

    n_step = np.ceil(n_times / tstep.astype(float)).astype(int)
    n_freq = wsize // 2 + 1
    shape = (-1, n_freq, n_step)
    n_coefs = (n_step * n_freq).astype(int)
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)

    logger.info('Computing and normalizing alpha_max ...')
    alpha_max = compute_alpha_max(
        G, M, phi, alpha_space, alpha_time, n_orient)
    alpha_max *= 0.01
    G /= alpha_max

    logger.info("Using block coordinate descent with active set")
    if n_orient == 1:
        lc = np.sum(G * G, axis=0)
    else:
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_tmp = G[:, (j * n_orient):((j + 1) * n_orient)]
            lc[j] = linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)

    X, Z, active_set, E, timeline = _tf_mixed_norm_solver_bcd_active_set(
        M, G, alpha_space, alpha_time, lc, phi, phiT, Z_init=None,
        w_space=w_space, w_time=w_time, wsize=wsize, tstep=tstep,
        n_orient=n_orient, maxit=maxit, tol=tol,
        log_objective=log_objective, verbose=None)

    if (active_set.sum() > 0) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    X /= alpha_max

    return X, active_set, E


@verbose
def iterative_tf_mixed_norm_solver(M, G, alpha_space, alpha_time, n_tfmxne_iter,
                                   wsize=64, tstep=4, maxit=3000,
                                   tol=1e-8, verbose=None, debias=True,
                                   log_objective=True, n_orient=1):
    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape
    n_positions = n_sources // n_orient

    n_step = np.ceil(n_times / tstep.astype(float)).astype(int)
    n_freq = wsize // 2 + 1
    n_coefs = (n_step * n_freq).astype(int)
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freq, n_step, n_times)

    logger.info('Computing and normalizing alpha_max ...')
    alpha_max = compute_alpha_max(
        G, M, phi, alpha_space, alpha_time, n_orient)
    alpha_max *= 0.01
    G /= alpha_max

    logger.info("Using block coordinate descent with active set")
    if n_orient == 1:
        lc = np.sum(G * G, axis=0)
    else:
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_tmp = G[:, (j * n_orient):((j + 1) * n_orient)]
            lc[j] = linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)

    if n_tfmxne_iter < 1:
        raise Exception('TF-MxNE has to be computed at least 1 time.')

    # lp-norm
    g_space = lambda w: np.sqrt(np.sqrt(stft_norm2_multidict(w).reshape(
        -1, n_orient).sum(axis=1)))
    g_space_prime = lambda w: 2. * np.sqrt(np.sqrt(stft_norm2_multidict(
        w).reshape(-1, n_orient).sum(axis=1)))

    g_time = lambda w: np.sqrt(np.sqrt(np.sum((np.abs(w) ** 2.).reshape(
        (n_orient, -1), order='F'), axis=0)).reshape((-1, w.shape[1]),
        order='F'))
    g_time_prime = lambda w: 2. * np.sqrt(np.sqrt(np.sum((np.abs(
        w) ** 2.).reshape((n_orient, -1), order='F'), axis=0)).reshape((
        -1, w.shape[1]), order='F'))

    E = list()

    active_set = np.ones(n_sources, dtype=np.bool)
    Z = np.zeros((n_sources, n_coefs.sum()), dtype=np.complex)
    X = np.zeros((n_sources, n_times))

    for k in range(n_tfmxne_iter):
        active_set0 = active_set.copy()

        if k == 0:
            w_space = None
            w_time = None
        else:
            w_space = 1. / g_space_prime(Z.copy())
            w_time = 1. / g_time_prime(Z.copy())
            w_time[g_time_prime(Z.copy()) == 0.] = 1e100

        X, Z, active_set_, E_ = _tf_mixed_norm_solver_bcd_active_set(
            M, G[:, active_set], alpha_space, alpha_time,
            lc[active_set[::n_orient]], phi, phiT, Z_init=Z, w_space=w_space,
            w_time=w_time, wsize=wsize, tstep=tstep, n_orient=n_orient,
            maxit=maxit, tol=tol, log_objective=log_objective, verbose=0)

        active_set[active_set] = active_set_.copy()

        if active_set.sum() > 0:
            _norm_l21_tf = np.sum(g_space(Z.copy()))
            _norm_l1_tf = np.sum(g_time(Z.copy()))
            p_obj = (0.5 * linalg.norm(M - np.dot(G[:, active_set],  X),
                     'fro') ** 2. + alpha_space * _norm_l21_tf +
                     alpha_time * _norm_l1_tf)
            E.append(p_obj)
            logger.info('Iteration %d: as_size=%d, E=%f' % (
                        k + 1, active_set.sum() / n_orient, p_obj))

            # Check convergence
            if k > 0:
                if np.array_equal(active_set0, active_set):
                    max_diff = np.max(np.abs(E[-2] - E[-1]) / E[-2])
                    print(max_diff)
                    if (max_diff < tol):
                        print('irTFMxNE: Convergence reached after %d reweightings!' % k)
                        break
        else:
            p_obj = 0.5 * linalg.norm(M) ** 2.
            E.append(p_obj)
            logger.info('Iteration %d: as_size=%d, E=%f' % (
                        k + 1, active_set.sum() / n_orient, p_obj))
            break

    if (active_set.sum() > 0) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]
        Z *= bias[:, np.newaxis]

    if (active_set.sum() > 0):
        X /= alpha_max
        Z /= alpha_max

    return X, active_set, E
