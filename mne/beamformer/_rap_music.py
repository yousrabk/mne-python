"""Compute a Recursively Applied and Projected MUltiple
Signal Classification (RAP-MUSIC).
"""

# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np
from scipy import linalg

from ..io.pick import pick_channels_evoked
from ..utils import logger, verbose
from ..dipole import Dipole
from ..source_estimate import SourceEstimate
from ..minimum_norm.inverse import combine_xyz
from ..io.proj import deactivate_proj
from ..beamformer._lcmv import _prepare_beamformer_input, _setup_picks
from ..forward import is_fixed_orient, convert_forward_solution
from ..cov import compute_whitener


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.05f}'.format(x, y, z)


def _make_sparse_stc(X, active_set, forward, tmin, tstep,
                     active_is_idx=False, verbose=None):
    if not is_fixed_orient(forward):
        logger.info('combining the current components...')
        X = combine_xyz(X)

    if not active_is_idx:
        active_idx = np.where(active_set)[0]
    else:
        active_idx = active_set

    n_dip_per_pos = 1 if is_fixed_orient(forward) else 3
    if n_dip_per_pos > 1:
        active_idx = np.unique(active_idx // n_dip_per_pos)

    src = forward['src']

    n_lh_points = len(src[0]['vertno'])
    lh_vertno = src[0]['vertno'][active_idx[active_idx < n_lh_points]]
    rh_vertno = src[1]['vertno'][active_idx[active_idx >= n_lh_points]
                                 - n_lh_points]
    vertices = [lh_vertno, rh_vertno]
    stc = SourceEstimate(X, vertices=vertices, tmin=tmin, tstep=tstep)
    return stc


def _compute_residual(forward, evoked, X, active_set, info):
    sel = [forward['sol']['row_names'].index(c) for c in info['ch_names']]
    residual = evoked.copy()
    residual = pick_channels_evoked(residual, include=info['ch_names'])
    r_tmp = residual.copy()
    r_tmp.data = np.dot(forward['sol']['data'][sel, :][:, active_set], X)
    if evoked.proj:
        active_projs = list()
        non_active_projs = list()
        for p in evoked.info['projs']:
            if p['active']:
                active_projs.append(p)
            else:
                non_active_projs.append(p)
        r_tmp.info['projs'] = deactivate_proj(active_projs, copy=True)
        r_tmp.apply_proj()
        r_tmp.add_proj(deepcopy(non_active_projs), remove_existing=False)
    residual.data -= r_tmp.data

    return residual


def _compute_2dip(G, phi_sig_proj, n_orient, subcorr_max, source_idx):
    for idx_1 in range(G.shape[1] // n_orient):
        for idx_2 in range(idx_1, G.shape[1] // n_orient):
            idx_k1 = slice(n_orient * idx_1, n_orient * (idx_1 + 1))
            idx_k2 = slice(n_orient * idx_2, n_orient * (idx_2 + 1))
            idx_k = np.r_[idx_k1, idx_k2]
            Gk = G[:, idx_k]
            subcorr = _compute_subcorr_nodir(Gk, phi_sig_proj, n_orient)
            if subcorr > subcorr_max:
                subcorr_max = subcorr
                source_idx = np.array([idx_1, idx_2])
    return subcorr_max, source_idx


def _compute_1dip(G, phi_sig_proj, n_orient, subcorr_max):
    for idx_1 in range(G.shape[1] // n_orient):
        idx_k = slice(n_orient * idx_1, n_orient * (idx_1 + 1))
        Gk = G[:, idx_k]
        subcorr = _compute_subcorr_nodir(Gk, phi_sig_proj, n_orient)
        if subcorr > subcorr_max:
            subcorr_max = subcorr
            source_idx = np.array([idx_1, idx_1])
    return subcorr_max, source_idx


def _estimate_subspace(data, thresh):
    eig_values, eig_vectors = linalg.eigh(np.cov(data))
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    rank_sig = np.sum(eig_values > thresh)
    print('rank of data = %d' % rank_sig)
    phi_sig = eig_vectors[:, :rank_sig]
    proj_data = np.dot(np.dot(phi_sig, phi_sig.T), data)
    return phi_sig, proj_data


def _seperate_sources(sol, source_idx, oris, n_sources, n_orient):
    n_dipoles_final = len(np.unique(np.ravel(source_idx)))
    oris_final = np.full((n_dipoles_final, 3), None)
    sol_final = np.empty((n_dipoles_final, sol.shape[1]))
    source_idx_final = np.empty(n_dipoles_final, dtype=int)
    active_set = np.zeros(n_sources, dtype=np.bool)
    i = 0
    for k in range(sol.shape[0]):
        if source_idx[k][0] == source_idx[k][1]:
            source_idx_final[i] = source_idx[k][0]
            if n_orient == 1:
                sol_final[i] = sol[k] * oris[k][0]
            else:
                sol_final[i] = sol[k]
                oris_final[i] = oris[k][:3]
                if oris_final[i][-1] < 0:
                    oris_final[i] *= -1.
                    sol_final[i] *= -1.
            active_set[n_orient * source_idx_final[i]:
                       n_orient * (source_idx_final[i] + 1)] = True
            i += 1
        else:
            source_idx_final[i] = source_idx[k][0]
            source_idx_final[i + 1] = source_idx[k][1]
            if n_orient == 1:
                sol_final[i] = sol[k] * oris[k][0]
                sol_final[i + 1] = sol[k] * oris[k][1]
            else:
                norm_o = linalg.norm(oris[k][:3])
                sol_final[i] = sol[k] * norm_o
                oris_final[i] = oris[k][:3] / norm_o
                if oris_final[i][-1] < 0:
                    oris_final[i] *= -1.
                    sol_final[i] *= -1.
                norm_o = linalg.norm(oris[k][3:])
                sol_final[i + 1] = sol[k] * norm_o
                oris_final[i + 1] = oris[k][3:] / norm_o
                if oris_final[i + 1][-1] < 0:
                    oris_final[i + 1] *= -1.
                    sol_final[i + 1] *= -1.
            active_set[n_orient * source_idx_final[i]:
                       n_orient * (source_idx_final[i] + 1)] = True
            active_set[n_orient * source_idx_final[i + 1]:
                       n_orient * (source_idx_final[i + 1] + 1)] = True
            i += 2

    idx_sort = np.argsort(source_idx_final.ravel())
    source_idx_final = source_idx_final[idx_sort]
    sol_final = sol_final[idx_sort]
    oris_final = oris_final[idx_sort]
    return sol_final, source_idx_final, oris_final, active_set


def _apply_rap_music(data, G, n_orient, n_dipoles=2, noise_variance=1.0,
                     corr_threshold=0.95, use_2dip=False):
    """RAP-MUSIC for evoked data

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        Evoked data.
    G : array, shape (n_channels, n_sources * n_orient)
        Gain matrix.
    n_orient : int
        Number of dipoles per locations (typically 1 or 3)
    n_dipoles : int
        The number of dipoles to estimate. The default value is 2.
    noise_variance : float
        Noise variance, used for wstimating the signal subspace.
    corr_threshold : float in [0, 1]
        Minimum subspace correlation to accept an IT as a true source.
    use_2dip : bool
        If True, single- and 2-dipole topographies are used.

    Returns
    -------
    sol : array | None
        Source time courses.
    source_idx_final : array | None
        Indices of active sources.
    oris_final : array | None
        Source orientations of active sources.
    active_set : array of bool
        Mask of active sources
    explained_data : array | None
        Data explained by the dipoles using a least square fitting with the
        selected active dipoles and their estimated orientation.
    max_corr : array
        Subspace correlations.
    """

    phi_sig, proj_data = _estimate_subspace(data, noise_variance)
    if phi_sig.shape[1] < n_dipoles:
        n_dipoles = phi_sig.shape[1]

    n_channels, n_sources = G.shape
    A = np.empty((n_channels, n_dipoles))
    source_idx_final = np.empty((n_dipoles, 2), dtype=int)
    oris_final = np.full((n_dipoles, 6), None)
    max_corr = np.empty(n_dipoles)
    active_set = np.zeros(n_sources, dtype=np.bool)

    G_proj = G.copy()
    phi_sig_proj = phi_sig.copy()

    for k in range(n_dipoles):
        subcorr_max = -1
        # compute subspace correlations with single and 2-dipole ITs
        subcorr_max, source_idx = _compute_1dip(
            G_proj, phi_sig_proj, n_orient, subcorr_max)
        idx_k = slice(n_orient * source_idx[0], n_orient * (source_idx[0] + 1))
        if use_2dip and (subcorr_max < corr_threshold):
            subcorr_max_2dip, source_idx_2dip = _compute_2dip(
                G_proj, phi_sig_proj, n_orient, subcorr_max, source_idx)
            if subcorr_max_2dip >= corr_threshold:
                subcorr_max = subcorr_max_2dip
                source_idx = source_idx_2dip
                idx_k1 = slice(n_orient * source_idx[0],
                               n_orient * (source_idx[0] + 1))
                idx_k2 = slice(n_orient * source_idx[1],
                               n_orient * (source_idx[1] + 1))
                idx_k = np.r_[idx_k1, idx_k2]
        if subcorr_max < corr_threshold:
            n_dipoles = k
            oris_final = oris_final[:n_dipoles, :]
            source_idx_final = source_idx_final[:n_dipoles, :]
            max_corr[k] = subcorr_max
            max_corr = max_corr[:(n_dipoles + 1)]
            A = A[:, :n_dipoles]
            logger.info("%d independant topographies found " % n_dipoles)
            break
        else:
            # compute source orientations and new projection operator
            _, ori = _compute_subcorr(G_proj[:, idx_k], phi_sig_proj, n_orient)
            max_corr[k] = subcorr_max
            oris_final[k][:len(ori)] = ori
            source_idx_final[k] = source_idx
            A[:, k] = np.dot(G[:, idx_k], ori)
            projection = _compute_proj(A[:, :(k + 1)])
            G_proj = np.dot(projection, G)
            phi_sig_proj = np.dot(projection, phi_sig)
            if source_idx_final[k][0] == source_idx_final[k][1]:
                logger.info("source %d found: p1 = %d, corr = %0.3f" % (
                    k + 1, source_idx_final[k][0], max_corr[k]))
            else:
                logger.info("source %d found: p1 = %d, p2 = %d, corr = %0.3f"
                            % (k + 1, source_idx_final[k][0],
                               source_idx_final[k][1], max_corr[k]))
    if n_dipoles == 0:
        sol_final, source_idx_final, oris_final = None, None, None
        explained_data = None
    else:
        sol = linalg.lstsq(A, data)[0]
        explained_data = np.dot(A, sol)
        sol_final, source_idx_final, oris_final, active_set = \
            _seperate_sources(sol, source_idx_final, oris_final,
                              G.shape[1], n_orient)

    return (sol_final, source_idx_final, oris_final, active_set,
            explained_data, max_corr)


def _make_dipoles(times, poss, oris, sol, gof):
    """Instanciates a list of Dipoles

    Parameters
    ----------
    times : array, shape (n_times,)
        The time instants.
    poss : array, shape (n_dipoles, 3)
        The dipoles' positions.
    oris : array, shape (n_dipoles, 3)
        The dipoles' orientations.
    sol : array, shape (n_times,)
        The dipoles' amplitudes over time.
    gof : array, shape (n_times,)
        The goodness of fit of the dipoles.
        Shared between all dipoles.

    Returns
    -------
    dipoles : list
        The list of Dipole instances.
    """
    amplitude = sol * 1e9
    oris = np.array(oris)

    dipoles = []
    for i_dip in range(poss.shape[0]):
        i_pos = poss[i_dip][np.newaxis, :].repeat(len(times), axis=0)
        i_ori = oris[i_dip][np.newaxis, :].repeat(len(times), axis=0)
        dipoles.append(Dipole(times, i_pos, amplitude[i_dip],
                              i_ori, gof))

    return dipoles


def _compute_subcorr_nodir(G, phi_sig, n_orient):
    """ Compute the subspace correlation
    """
    Ug, Sg, _ = linalg.svd(G, full_matrices=False)
    rankSg = np.linalg.matrix_rank(np.diag(Sg))
    Sg = Sg[:rankSg]
    Ug = Ug[:, :rankSg]
    tmp = np.dot(Ug.conjugate().T, phi_sig)
    if tmp.shape[1] > tmp.shape[0]:
        Sc = linalg.svd(tmp.conjugate().T, full_matrices=False,
                        compute_uv=False)
    else:
        Sc = linalg.svd(tmp, full_matrices=False, compute_uv=False)
    return Sc[0]


def _compute_subcorr(G, phi_sig, n_orient):
    """ Compute the subspace correlation and orientation of the IT.
    """
    Ug, Sg, Vg = linalg.svd(G, full_matrices=False)
    tmp = np.dot(Ug.conjugate().T, phi_sig)
    if tmp.shape[1] > tmp.shape[0]:
        Uc, Sc, Vc = linalg.svd(tmp.conjugate().T, full_matrices=False)
        Uc = Vc.T
    else:
        Uc, Sc, Vc = linalg.svd(tmp, full_matrices=False)
    X = np.dot(np.dot(Vg.T, np.diag(1. / Sg)), Uc)
    return Sc[0], X[:, 0] / linalg.norm(X[:, 0])


def _compute_proj(A):
    """ Compute the orthogonal projection operation for
    a manifold vector A.
    """
    U, S, _ = linalg.svd(A, full_matrices=False)
    return np.identity(A.shape[0]) - np.dot(U, U.T.conjugate())


@verbose
def rap_music(evoked, forward, noise_cov, n_dipoles=3, return_residual=False,
              picks=None, noise_variance=1.0, corr_threshold=0.95,
              use_2dip=False, verbose=None):
    """RAP-MUSIC source localization method.

    Compute Recursively Applied and Projected MUltiple SIgnal Classification
    (RAP-MUSIC) on evoked data.

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked data to localize.
    forward : instance of Forward
        Forward operator.
    noise_cov : instance of Covariance
        The noise covariance.
    n_dipoles : int | 'auto'
        If int, the number of dipoles to look for. If 'auto' the number
        of dipoles is obtained by thresholding the data covariance
        after whitening. Threshold on eigen values is set to 3.
    return_residual : bool
        If True, the residual is returned as an Evoked instance.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    noise_variance : float
        Noise variance, used for wstimating the signal subspace.
    corr_threshold : float in [0, 1]
        Minimum subspace correlation to accept an IT as a true source.
    use_2dip : bool
        If True, single- and 2-dipole topographies are used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : instance of SourceEstimate | None
        The source time courses as a sparse stc file.
    dipoles : list of instance of Dipole | None
        The dipole fits.
    residual : instance of Evoked | None
        The residual a.k.a. data not explained by the dipoles.
        Only returned if return_residual is True.

    Notes
    -----
    The references are:

        J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
        applied and projected (RAP) MUSIC. Signal Processing, IEEE Trans. 47, 2
        (February 1999), 332-340.
        DOI=10.1109/78.740118
        http://dx.doi.org/10.1109/78.740118

        Mosher, J.C.; Leahy, R.M., EEG and MEG source localization using
        recursively applied (RAP) MUSIC, Signals, Systems and Computers, 1996.
        pp.1201,1207 vol.2, 3-6 Nov. 1996
        doi: 10.1109/ACSSC.1996.599135
        http://dx.doi.org/10.1109/ACSSC.1996.599135

    .. versionadded:: 0.9.0
    """

    if (corr_threshold < 0.0) or (corr_threshold > 1.0):
        raise ValueError('Invalid subspace correlations threshold. '
                         'Requires 0.0 <= corr_threshold <= 1.0. '
                         'Got corr_threshold = %f.' % corr_threshold)

    n_orient = 1 if is_fixed_orient(forward) else 3
    if forward['surf_ori'] is False:
        fwd = convert_forward_solution(
            forward, surf_ori=True, force_fixed=False, copy=True)
    else:
        fwd = deepcopy(forward)

    picks = _setup_picks(picks, evoked.info, fwd, noise_cov)

    times = evoked.times
    data = evoked.data[picks]

    is_free_ori, ch_names, proj, vertno, G = _prepare_beamformer_input(
        evoked.info, fwd, label=None, picks=picks, pick_ori=None)

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, evoked.info, picks,
                                   nave=evoked.nave)
    if evoked.info['projs']:
        whitener = np.dot(whitener, proj)

    # whiten the leadfield and the data
    G = np.dot(whitener, G)
    data = np.dot(whitener, data)

    (sol_final, source_idx_final, oris_final, active_set, explained_data,
        max_corr) = _apply_rap_music(
        data, G, n_orient, n_dipoles=n_dipoles, noise_variance=noise_variance,
        corr_threshold=corr_threshold)

    if sol_final is not None:
        poss_final = np.empty_like(oris_final)
        for i in range(sol_final.shape[0]):
            if n_orient == 1:
                oris_final[i] = forward['source_nn'][source_idx_final[i]]
            poss_final[i] = forward['source_rr'][source_idx_final[i]]

        if n_orient == 1:
            X = sol_final.copy()
        else:
            ori_mat = linalg.block_diag(*oris_final).T
            X = np.dot(ori_mat, sol_final)

        if return_residual:
            residual = _compute_residual(
                fwd, evoked, X, active_set, evoked.info)

        gof = 1. - (linalg.norm(data - explained_data)
                    / linalg.norm(data)) ** 2
        gof = np.sqrt(gof)

        # rotate source orientation to xyz
        if n_orient == 3:
            for i, idx_k in enumerate(source_idx_final):
                idx = slice(n_orient * idx_k, n_orient * (idx_k + 1))
                oris_final[i] = np.dot(fwd['source_nn'][idx].T, oris_final[i])

        dipoles = _make_dipoles(times, poss_final, oris_final, sol_final, gof)

    else:
        dipoles = []
        X = np.zeros((0, len(times)))
        active_set = np.zeros(G.shape[1], dtype=np.bool)
        if return_residual:
            residual = evoked.copy()
        logger.info("No independant topographies found")
        logger.info("maximum correlation = %.2f < %.2f!" % (
            max_corr[0], corr_threshold))

    stc = _make_sparse_stc(X, active_set, fwd, times[0], times[1] - times[0])

    if return_residual:
        return stc, dipoles, residual
    else:
        return stc, dipoles
