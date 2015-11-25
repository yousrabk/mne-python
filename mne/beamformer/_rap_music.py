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

import matplotlib.pyplot as plt

from mne.io.pick import pick_channels_evoked
from mne.utils import logger, verbose
from mne.dipole import Dipole
from mne.beamformer._lcmv import _prepare_beamformer_input, _setup_picks
from mne.forward import is_fixed_orient, convert_forward_solution

from mne.cov import compute_whitener


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


def _apply_rap_music(data, G, n_orient, n_dipoles=2):
    """RAP-MUSIC for evoked data

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        Evoked data.
    info : dict
        Measurement info.
    times : array
        Times.
    forward : instance of Forward
        Forward operator.
    noise_cov : instance of Covariance
        The noise covariance.
    n_dipoles : int
        The number of dipoles to estimate. The default value is 2.
    picks : array-like of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    return_explained_data : bool
        If True, the explained data is returned as an array.

    Returns
    -------
    dipoles : list of instances of Dipole
        The dipole fits.
    explained_data : array | None
        Data explained by the dipoles using a least square fitting with the
        selected active dipoles and their estimated orientation.
        Computed only if return_explained_data is True.
    """

    # eig_values, eig_vectors = linalg.eigh(np.cov(data))
    # rank_sig = np.sum(eig_values > 3.0)
    # print('rank of data = %d' % rank_sig)
    # phi_sig = eig_vectors[:, -rank_sig:]

    U, s, V = linalg.svd(data, full_matrices=False)
    plt.figure(), plt.plot(s ** 2.), plt.show()
    # rank_sig = np.sum((s ** 2.) > 3.0)
    rank_sig = int(raw_input("Estimated data rank: "))
    print('rank of data = %d' % rank_sig)
    phi_sig = U[:, :rank_sig]
    if rank_sig < n_dipoles:
        n_dipoles = rank_sig

    s2 = s.copy()
    s2[rank_sig:] = 0
    data_test = np.dot(U, np.dot(np.diag(s2), V))
    plt.figure(), plt.plot(data.T), plt.show()
    plt.figure(), plt.plot(data_test.T), plt.show()

    n_channels = G.shape[0]
    A = np.empty((n_channels, n_dipoles))
    source_idx_final = np.empty((n_dipoles, 2))
    oris_final = np.empty((n_dipoles, 6))
    max_corr = np.empty(n_dipoles)

    G_proj = G.copy()
    phi_sig_proj = phi_sig.copy()

    corr_threshold = 0.95

    for k in range(n_dipoles):
        subcorr_max = -1.
        for idx_1 in range(G.shape[1] // n_orient):
            idx_k = slice(n_orient * idx_1, n_orient * (idx_1 + 1))
            Gk = G_proj[:, idx_k]
            subcorr = _compute_subcorr_nodir(Gk, phi_sig_proj, n_orient)
            if subcorr > subcorr_max:
                subcorr_max = subcorr.copy()
                source_idx = np.array([idx_1, idx_1])
            idx_k = slice(n_orient * source_idx[0],
                          n_orient * (source_idx[0] + 1))
            Gk = G_proj[:, idx_k]
            _, ori = _compute_subcorr(Gk, phi_sig_proj, n_orient)
            source_ori = ori.copy()
            if n_orient == 3:
                if source_ori[2] < 0:
                    source_ori *= -1
        if subcorr_max >= corr_threshold:
            idx_k = slice(n_orient * source_idx[0],
                          n_orient * (source_idx[0] + 1))
            Ak = G[:, idx_k]
            Ak = np.dot(Ak, source_ori[:, None])
            A[:, k] = Ak.ravel()
            max_corr[k] = subcorr_max
            if n_orient == 3:
                oris_final[k] = np.r_[source_ori, np.array([None, None, None])]
            else:
                oris_final[k] = np.r_[source_ori,
                                      np.array([None, None, None, None, None])]
            source_idx_final[k] = source_idx
        else:
            subcorr_mat = np.zeros((G.shape[1] // n_orient,
                                    G.shape[1] // n_orient))
            for idx_1 in range(G.shape[1] // n_orient):
                for idx_2 in range(idx_1 + 1, G.shape[1] // n_orient):
                    idx_k1 = slice(n_orient * idx_1, n_orient * (idx_1 + 1))
                    idx_k2 = slice(n_orient * idx_2, n_orient * (idx_2 + 1))
                    idx_k = np.r_[idx_k1, idx_k2]
                    Gk = G_proj[:, idx_k]
                    subcorr = _compute_subcorr_nodir(Gk, phi_sig_proj,
                                                     n_orient)
                    subcorr_mat[idx_1, idx_2] = subcorr.copy()
                    if subcorr > subcorr_max:
                        subcorr_max = subcorr.copy()
                        source_idx = np.array([idx_1, idx_2])
                        _, ori = _compute_subcorr(Gk, phi_sig_proj, n_orient)
                        source_ori = ori.copy()
                        if n_orient == 3:
                            if source_ori[2] < 0:
                                source_ori[:3] *= -1
                            if source_ori[-1] < 0:
                                source_ori[3:] *= -1
            fig, ax = plt.subplots()
            im = ax.imshow(subcorr_mat, interpolation='none')
            ax.format_coord = Formatter(im)
            plt.show()
            if subcorr_max < corr_threshold:
                n_dipoles = k
                oris_final = oris_final[:n_dipoles, :]
                source_idx_final = source_idx_final[:n_dipoles, :]
                max_corr[k] = subcorr_max
                max_corr = max_corr[:(n_dipoles + 1)]
                logger.info("%d independant topographies found " % n_dipoles)
                break
            else:
                max_corr[k] = subcorr_max
                idx_k1 = slice(n_orient * source_idx[0],
                               n_orient * (source_idx[0] + 1))
                idx_k2 = slice(n_orient * source_idx[1],
                               n_orient * (source_idx[1] + 1))
                idx_k = np.r_[idx_k1, idx_k2]
                Ak = G[:, idx_k]
                Ak = np.dot(Ak, source_ori[:, None])
                A[:, k] = Ak.ravel()
                if n_orient == 3:
                    oris_final[k] = source_ori
                else:
                    oris_final[k] = np.r_[source_ori,
                                          np.array([None, None, None, None])]
                source_idx_final[k] = source_idx

        if source_idx_final[k][0] == source_idx_final[k][1]:
            logger.info("source %s found: p1 = %s" % (
                k + 1, source_idx_final[k][0]))
        else:
            logger.info("source %s found: p1 = %s, p2 = %s" % (
                k + 1, source_idx_final[k][0], source_idx_final[k][1]))

        projection = _compute_proj(A[:, :(k + 1)])
        G_proj = np.dot(projection, G)
        phi_sig_proj = np.dot(projection, phi_sig)

    if n_dipoles:
        sol = linalg.lstsq(A[:, :(n_dipoles + 1)], data)[0]
    else:
        sol = None

    return sol, source_idx_final, oris_final, max_corr


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


def _compute_subcorr_old(G, phi_sig):
    """ Compute the subspace correlation
    """
    Ug, Sg, Vg = linalg.svd(G, full_matrices=False)
    tmp = np.dot(Ug.T.conjugate(), phi_sig)
    Uc, Sc, Vc = linalg.svd(tmp, full_matrices=False)
    X = np.dot(np.dot(Vg.T, np.diag(1. / Sg)), Uc)  # subcorr
    return Sc[0], X[:, 0] / linalg.norm(X[:, 0])


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
    """ Compute the subspace correlation
    """
    Ug, Sg, Vg = linalg.svd(G, full_matrices=False)
    rankSg = np.linalg.matrix_rank(np.diag(Sg))
    tmp = np.dot(Ug.conjugate().T, phi_sig)
    if tmp.shape[1] > tmp.shape[0]:
        Uc, Sc, Vc = linalg.svd(tmp.conjugate().T, full_matrices=False)
        Uc = Vc.T
    else:
        Uc, Sc, Vc = linalg.svd(tmp, full_matrices=False)
    X = np.dot(np.dot(Vg.T, np.diag(1. / Sg[:rankSg])), Uc)
    return Sc[0], X[:, 0] / linalg.norm(X[:, 0])


def _compute_proj(A):
    """ Compute the orthogonal projection operation for
    a manifold vector A.
    """
    U, _, _ = linalg.svd(A, full_matrices=False)
    return np.identity(A.shape[0]) - np.dot(U, U.T.conjugate())


@verbose
def rap_music(evoked, forward, noise_cov, n_dipoles=3, pca=False,
              return_residual=False, picks=None, return_explained_data=False,
              verbose=None):
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    dipoles : list of instance of Dipole
        The dipole fits.
    residual : instance of Evoked
        The residual a.k.a. data not explained by the dipoles.
        Only returned if return_residual is True.

    Notes
    -----
    The references are:

        J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
        applied and projected (RAP) MUSIC. Signal Processing, IEEE Trans. 47, 2
        (February 1999), 332-340.
        DOI=10.1109/78.740118 http://dx.doi.org/10.1109/78.740118

        Mosher, J.C.; Leahy, R.M., EEG and MEG source localization using
        recursively applied (RAP) MUSIC, Signals, Systems and Computers, 1996.
        pp.1201,1207 vol.2, 3-6 Nov. 1996
        doi: 10.1109/ACSSC.1996.599135

    .. versionadded:: 0.9.0
    """

    n_orient = 1 if is_fixed_orient(forward) else 3
    if n_orient == 3:
        fwd = convert_forward_solution(
            forward, surf_ori=True, force_fixed=False, copy=True)
    else:
        fwd = deepcopy(forward)

    picks = _setup_picks(picks, evoked.info, forward, noise_cov)

    times = evoked.times
    data = evoked.data[picks]

    is_free_ori, ch_names, proj, vertno, G = _prepare_beamformer_input(
        evoked.info, fwd, label=None, picks=picks, pick_ori=None)

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, evoked.info, picks)

    if evoked.info['projs']:
        whitener = np.dot(whitener, proj)

    # whiten the leadfield and the data
    G = np.dot(whitener, G)
    gain = G.copy()
    data = np.dot(whitener, data)

    sol, source_idx, oris, max_corr = _apply_rap_music(
        data, G, n_orient, n_dipoles=n_dipoles)

    if sol is not None:
        print max_corr
        n_dipoles_final = len(np.unique(np.ravel(source_idx)))
        oris_final = np.empty((n_dipoles_final, 3))
        poss_final = np.empty((n_dipoles_final, 3))
        sol_final = np.empty((n_dipoles_final, sol.shape[1]))
        source_idx_final = np.empty(n_dipoles_final)
        active_set = np.zeros(G.shape[1], dtype=np.bool)
        i = 0
        for k in range(sol.shape[0]):
            if source_idx[k][0] == source_idx[k][1]:
                source_idx_final[i] = source_idx[k][0]
                if n_orient == 1:
                    sol_final[i] = sol[k] * oris[k][0]
                    oris_final[i] = forward['source_nn'][source_idx[k][0]]
                else:
                    sol_final[i] = sol[k]
                    oris_final[i] = oris[k][:3][None, :]
                    oris_final[i] = \
                        np.dot(forward['source_nn'][source_idx[k][0]],
                               oris_final[i])
                poss_final[i] = forward['source_rr'][source_idx[k][0]]
                active_set[n_orient * source_idx[k][0]:
                           n_orient * (source_idx[k][0] + 1)] = True
                i += 1
            else:
                source_idx_final[i] = source_idx[k][0]
                source_idx_final[i + 1] = source_idx[k][1]
                if n_orient == 1:
                    sol_final[i] = sol[k] * oris[k][0]
                    oris_final[i] = forward['source_nn'][source_idx[k][0]]
                    sol_final[i + 1] = sol[k] * oris[k][1]
                    oris_final[i + 1] = forward['source_nn'][source_idx[k][1]]
                else:
                    norm_o = linalg.norm(oris[k][:3])
                    sol_final[i] = sol[k] * norm_o
                    oris_final[i] = oris[k][:3][None, :] / norm_o
                    oris_final[i] = \
                        np.dot(forward['source_nn'][source_idx[k][0]],
                               oris_final[i])
                    norm_o = linalg.norm(oris[k][3:])
                    sol_final[i + 1] = sol[k] * norm_o
                    oris_final[i + 1] = oris[k][3:][None, :] / norm_o
                    oris_final[i + 1] = np.dot(
                        forward['source_nn'][source_idx[k][1]],
                        oris_final[i + 1])
                poss_final[i] = forward['source_rr'][source_idx[k][0]]
                poss_final[i + 1] = forward['source_rr'][source_idx[k][1]]
                active_set[n_orient * source_idx[k][0]:
                           n_orient * (source_idx[k][0] + 1)] = True
                active_set[n_orient * source_idx[k][1]:
                           n_orient * (source_idx[k][1] + 1)] = True
                i += 2

        idx_sort = np.argsort(source_idx_final.ravel())
        source_idx_final = source_idx_final[idx_sort]
        sol_final = sol_final[idx_sort]
        oris_final = oris_final[idx_sort]
        poss_final = poss_final[idx_sort]

        if n_orient == 1:
            X = sol_final.copy()
        else:
            ori_mat = linalg.block_diag(*oris_final).T
            X = np.dot(ori_mat, sol)

        from mxne_inverse import _make_sparse_stc
        stc = _make_sparse_stc(X, active_set, forward, times[0],
                               times[1] - times[0])

        explained_data = np.dot(gain[:, active_set], X)
        gof = (linalg.norm(np.dot(whitener, explained_data)) /
               linalg.norm(data))

        dipoles = _make_dipoles(times, poss_final, oris_final, sol_final, gof)

        if return_residual:
            residual = evoked.copy()
            selection = [evoked.info['ch_names'][p] for p in picks]
            residual = pick_channels_evoked(residual, include=selection)
            residual.data -= explained_data
            active_projs = [p for p in residual.info['projs'] if p['active']]
            for p in active_projs:
                p['active'] = False
            residual.add_proj(active_projs, remove_existing=True)
            residual.apply_proj()
    else:
        stc, dipoles, residual = None, None, None
        logger.info("No independant topographies found")
        logger.info("maximum correlation = %.2f < 0.95!" % max_corr[0])

    if return_residual:
        return stc, dipoles, residual
    else:
        return stc, dipoles
