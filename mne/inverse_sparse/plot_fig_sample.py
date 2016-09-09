import mne

from mne.datasets import sample
from mne.inverse_sparse import mixed_norm
from mne.viz import plot_sparse_source_estimates

import numpy as np

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read the evoked response and crop it
condition = 'Left Auditory'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(None, 0))
evoked.crop(tmin=0.05, tmax=0.15)  # select N100

# evoked.pick_types(meg='grad', eeg=False)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname, surf_ori=True,
                                    force_fixed=False)

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)

update_alpha = True
alpha = 20. * np.ones((forward['sol']['data'].shape[1]))
alpha = 50.
name = "estimate one hp"

stc, _ = mixed_norm(evoked, forward, noise_cov, alpha=alpha,
                    update_alpha=update_alpha, verbose=True, hp_iter=3)

plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.1, fig_name="%s (cond %s)"
                             % (name, condition), modes=['sphere'],
                             scale_factors=[1.])
