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
# evoked.crop(tmin=0.043, tmax=0.15)  # select N100
evoked.crop(tmin=0.058, tmax=0.14)  # select N100

evoked.pick_types(meg=True, eeg=False)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname, surf_ori=True,
                                    force_fixed=True)
src = forward['src']

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)

update_alpha = True
# alpha = 69.5 * np.ones((forward['sol']['data'].shape[1]))
alpha = 20 * np.ones((forward['sol']['data'].shape[1]))
# alpha = 70.
n_mxne_iter = 1
name = "estimate one hp per source"

stc, als = mixed_norm(evoked, forward, noise_cov, alpha=alpha,
                      update_alpha=update_alpha, verbose=True, hp_iter=10,
                      n_mxne_iter=n_mxne_iter)

plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.1, fig_name="%s (cond %s)"
                             % (name, condition), modes=['sphere'],
                             scale_factors=[1.])

# time_label = 'time=%0.2f ms'
# # clim = dict(kind='value', lims=[10e-9, 15e-9, 20e-9])
# brain = stc.plot('sample', 'inflated', 'lh', time_label=time_label,
#                  smoothing_steps=5, subjects_dir=subjects_dir)
# # brain.show_view('medial')
# brain.hide_colorbar()
# src_lh = mne.read_surface(subjects_dir + '/sample/surf/lh.inflated')[0]

# brain.set_data_time_index(22)
# brain.add_foci(src_lh[stc.vertices[0]], color='yellow',
#                scale_factor=1.)

# time_label = 'time=%0.2f ms'
# # clim = dict(kind='value', lims=[10e-9, 15e-9, 20e-9])
# brain = stc.plot('sample', 'inflated', 'rh', time_label=time_label,
#                  smoothing_steps=5, subjects_dir=subjects_dir)
# # brain.show_view('medial')
# brain.hide_colorbar()
# src_lh = mne.read_surface(subjects_dir + '/sample/surf/rh.inflated')[0]

# brain.set_data_time_index(34)
# brain.add_foci(src_lh[stc.vertices[1]], color='yellow',
#                scale_factor=1.)
