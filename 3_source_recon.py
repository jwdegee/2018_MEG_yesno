import os
import glob
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

from IPython import embed as shell

import pymeg
from pymeg import preprocessing as prep
from pymeg import source_reconstruction as sr
from pymeg import lcmv 

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

data_folder = '/home/jw/share/data/'
sr.set_fs_subjects_dir('/home/jw/share/data/fs_subjects/')

def get_highF_estimator(sf=600, decim=10):
    fois = np.arange(42, 162, 2)
    cycles = 0.4 * fois
    tb = 5 + 1
    return ('HF', fois, lcmv.get_power_estimator(fois, cycles, tb, sf=sf,
                                           decim=decim))

def get_lowF_estimator(sf=600, decim=10):
    fois = np.arange(1, 41, 1)
    cycles = 0.4 * fois
    tb = 1 + 1
    return ('LF', fois, lcmv.get_power_estimator(fois, cycles, tb, sf=sf,
                                            decim=decim))

def get_broadband_estimator():
    return ('BB', [-1], lambda x: x[:, np.newaxis, :])

# subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
# subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09',]
subjects = ['jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17',]
# subjects = ['jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30',]

# tfr:
for subj in subjects:
    for session in ["A", "B"]:
        
        runs = sorted([run.split('/')[-1] for run in glob.glob(os.path.join(data_folder, "raw", subj, session, "meg", "*.ds"))])
        center = int(np.floor(len(runs) / 2.0))
        
        raw_filename = os.path.join(data_folder, "raw", subj, session, "meg", runs[center])
        epochs_filename_stim = os.path.join(data_folder, "epochs", subj, session, '{}-epo.fif.gz'.format('stimlock'))
        epochs_filename_resp = os.path.join(data_folder, "epochs", subj, session, '{}-epo.fif.gz'.format('resplock'))
        trans_filename = os.path.join(data_folder, "transformation_matrix", '{}_{}-trans.fif'.format(subj, session))

        if os.path.isfile(epochs_filename_stim):
            
            # # make transformation matrix:
            # sr.make_trans(subj, raw_filename, epochs_filename, trans_filename)
            
            # load epochs:
            epochs_stim = mne.read_epochs(epochs_filename_stim)
            epochs_stim = epochs_stim.pick_channels([x for x in epochs_stim.ch_names if x.startswith('M')])

            epochs_resp = mne.read_epochs(epochs_filename_resp)
            epochs_resp = epochs_resp.pick_channels([x for x in epochs_resp.ch_names if x.startswith('M')])
            
            # baseline stuff:
            overlap = list(
                set(epochs_stim.events[:, 2]).intersection(
                set(epochs_resp.events[:, 2])))
            epochs_stim = epochs_stim[[str(l) for l in overlap]]
            epochs_resp = epochs_resp[[str(l) for l in overlap]]
            id_time = (-0.3 <= epochs_stim.times) & (epochs_stim.times <= -0.2)
            means = epochs_stim._data[:, :, id_time].mean(-1)
            epochs_stim._data = epochs_stim._data - means[:, :, np.newaxis]
            epochs_resp._data = epochs_resp._data - means[:, :, np.newaxis]

            # get cov:
            data_cov = lcmv.get_cov(epochs_stim, tmin=0, tmax=1)
            noise_cov = None

            # get lead field:
            forward, bem, source = sr.get_leadfield(
                                                    subject=subj, 
                                                    raw_filename=raw_filename, 
                                                    epochs_filename=epochs_filename_stim, 
                                                    trans_filename=trans_filename,
                                                    conductivity=(0.3, 0.006, 0.3)
                                                    )

            # get labels:
            labels = sr.get_labels(subj, ['*wang*.label', '*HCPMMP1*.label'])

            # do source level analysis:
            for tl, epochs in zip(['stimlock', 'resplock'], [epochs_stim, epochs_resp]):

                # shell()

                # meta_filename = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format(tl))
                sr_filename = os.path.join(data_folder, "source_level", 'average_{}_{}_{}-source.fif'.format(subj, session, tl))
                lcmv_filename = os.path.join(data_folder, "source_level", 'lcmv_{}_{}_{}-source.hdf'.format(subj, session, tl))
                
                # load meta data:
                # meta = pd.concat(prep.load_meta([meta_filename]))
                
                # estimator and accumulator:
                estimators = (#get_broadband_estimator(),
                            get_highF_estimator(),
                            get_lowF_estimator())
                accumulator = lcmv.AccumSR(sr_filename, 'F', 80)

                # do source reconstruction:
                source_epochs = lcmv.reconstruct(
                                                epochs=epochs,
                                                forward=forward,
                                                source=source,
                                                noise_cov=noise_cov,
                                                data_cov=data_cov,
                                                labels=labels,
                                                func=estimators,
                                                accumulator=None,
                                                first_all_vertices=False,
                                                debug=False)

                # save:
                source_epochs.to_hdf(lcmv_filename, 'epochs')
                # accumulator.save_averaged_sr()
                 
                