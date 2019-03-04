import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import mne
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from joblib import Parallel, delayed
from joblib import Memory

from IPython import embed as shell

import pymeg
from pymeg import preprocessing as prep
from pymeg import atlas_glasser
from pymeg import contrast_tfr

import jw_tools.myfuncs as myfuncs

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

memory = Memory(location=os.environ['PYMEG_CACHE_DIR'], verbose=0)

data_folder = '/media/external3/JW/meg/data/'
fig_folder = '/media/external3/JW/meg/figures/'

def select_subjects(subjects, conditions=[], cutoff=25):

    subjs = []
    sess = []
    counts = [[] for _ in conditions]
    for subj in subjects:
        # print(subj)
        # try:
        #     session = 'B'
        #     timelock = 'stimlock'
        #     meta_data_filename = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format(timelock))
        #     meta_data = pymeg.preprocessing.load_meta([meta_data_filename])[0]
        #     pupil_data = pd.read_csv(os.path.join(data_folder, 'pupil', '{}_{}_pupil_data.csv'.format(subj, session)))
        #     print('{} - {} = {}'.format(pupil_data.shape[0], meta_data.shape[0], pupil_data.shape[0]-meta_data.shape[0]))
        # except:
        #     pass
        print(subj)
        try:
            meta_a = load_meta_data(subj, 'A', 'stimlock', data_folder)
            a = True
        except:
            a = False
        try:
            meta_b = load_meta_data(subj, 'B', 'stimlock', data_folder)
            b = True
        except:
            b = False 
        if a&b:
            meta_data = pd.concat((meta_a, meta_b))
        elif a:
            meta_data = meta_a.copy()
        elif b:
            meta_data = meta_b.copy()

        for i, c in enumerate(conditions):                           
            counts[i].append(sum(meta_data[c]==1))
        subjs.append(subj)
    subjs = np.array(subjs)
    counts = np.array(counts)

    fig = plt.figure(figsize=(3,12))
    for i, c in enumerate(conditions):
        ax = fig.add_subplot(len(conditions),1,i+1)
        ax.hist(counts[i,:], bins=20)
        ax.axvline(cutoff, ls='--', color='r')
        plt.xlim(xmin=0)
        plt.xlabel('# trials')
        plt.title(c)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_folder, 'trial_counts.pdf'))
    
    subj_ind = counts.min(axis=0) > cutoff

    print('keep {} out of {} subjects'.format(sum(subj_ind), len(subjects)))

    return list(np.array(subjects)[subj_ind])

def load_meta_data(subj, session, timelock, data_folder):

    # load:
    meta_data_filename = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format(timelock))
    meta_data = pymeg.preprocessing.load_meta([meta_data_filename])[0].reset_index(drop=True)
    pupil_data = pd.read_csv(os.path.join(data_folder, 'pupil', '{}_{}_pupil_data.csv'.format(subj, session))).reset_index(drop=True)

    # shell()

    # merge the bastards:
    meta_row = 0
    keep = []
    for pupil_row in range(pupil_data.shape[0]):
        
        try:
            meta = np.array(meta_data.iloc[meta_row][['noise', 'stimulus', 'choice_a', 'confidence',]], dtype=int)
            pupil = np.array(pupil_data.iloc[pupil_row][['noise', 'stimulus', 'choice_a', 'confidence',]], dtype=int)

            rt_meta = float(meta_data.iloc[meta_row][['rt',]])
            rt_pupil = float(pupil_data.iloc[pupil_row][['rt',]])

            if ((meta == pupil).sum()==4) & (abs(rt_meta-rt_pupil)<0.1):
                meta_row += 1
                keep.append(pupil_row)
        except IndexError:
            continue
    pupil_data = pupil_data.iloc[keep].reset_index()

    assert meta_data.shape[0] == pupil_data.shape[0] 
    print(meta_data.iloc[-1]['rt']-pupil_data.iloc[-1]['rt'])

    # add some columns:
    meta_data['trial_nr'] = np.array(pupil_data['trial_nr'], dtype=int)
    meta_data['pupil_lp_b'] = np.array(pupil_data['pupil_lp_b'], dtype=float)
    meta_data['pupil_lp_d'] = np.array(pupil_data['pupil_lp_d'], dtype=float)
    meta_data['signal_contrast'] = np.array(pupil_data['signal_contrast'], dtype=float)
    meta_data['signal_contrast'] = (10**meta_data['signal_contrast']) * 100
    meta_data.loc[meta_data['stimulus']==0, 'signal_contrast'] = 0
    meta_data['rt'] = meta_data['rt'].astype(float)

    # regress out:
    for noise in [1,4]:
        ind = (meta_data['noise'] == noise)
        meta_data.loc[ind, 'pupil_lp_d'] = myfuncs.lin_regress_resid(np.array(meta_data.loc[ind, 'pupil_lp_d']),
                                        [
                                        np.array(meta_data.loc[ind, 'rt']),
                                        np.array(meta_data.loc[ind, 'signal_contrast']),
                                        np.array(meta_data.loc[ind, 'pupil_lp_b']),
                                        ])

    # split by pupil or rt:
    meta_data["bin_pupil"] = meta_data.groupby(['noise'])['pupil_lp_d'].apply(pd.qcut, q=2, labels=False)
    meta_data["bin_rt"] =  meta_data.groupby(['noise'])['rt'].apply(pd.qcut, q=2, labels=False)
    meta_data["pupil_h"] = meta_data["bin_pupil"] == 1
    meta_data["pupil_l"] = meta_data["bin_pupil"] == 0
    meta_data["rt_h"] = meta_data["bin_rt"] == 1
    meta_data["rt_l"] = meta_data["bin_rt"] == 0

    # add columns:
    meta_data["all"] = 1
    meta_data["left"] = (meta_data["resp_meg"] < 0).astype(int)
    meta_data["right"] = (meta_data["resp_meg"] > 0).astype(int)
    meta_data["hit"] = ((meta_data["stimulus"] == 1) & (meta_data["choice_a"] == 1)).astype(int)
    meta_data["fa"] = ((meta_data["stimulus"] == 0) & (meta_data["choice_a"] == 1)).astype(int)
    meta_data["miss"] = ((meta_data["stimulus"] == 1) & (meta_data["choice_a"] == 0)).astype(int)
    meta_data["cr"] = ((meta_data["stimulus"] == 0) & (meta_data["choice_a"] == 0)).astype(int)
    meta_data["conf_h"] = (meta_data.confidence > 1).astype(int)
    meta_data["conf_l"] = (meta_data.confidence <= 1).astype(int)
    meta_data["noise_fast"] = (meta_data.noise == 1).astype(int)
    meta_data["noise_slow"] = (meta_data.noise == 4).astype(int)
    meta_data["hit_rt_h"] = ((meta_data["hit"] == 1) & (meta_data["rt_h"] == 1)).astype(int)
    meta_data["hit_rt_l"] = ((meta_data["hit"] == 1) & (meta_data["rt_l"] == 1)).astype(int)
    meta_data["hit_pupil_h"] = ((meta_data["hit"] == 1) & (meta_data["pupil_h"] == 1)).astype(int)
    meta_data["hit_pupil_l"] = ((meta_data["hit"] == 1) & (meta_data["pupil_l"] == 1)).astype(int)
    meta_data["fa_rt_h"] = ((meta_data["fa"] == 1) & (meta_data["rt_h"] == 1)).astype(int)
    meta_data["fa_rt_l"] = ((meta_data["fa"] == 1) & (meta_data["rt_l"] == 1)).astype(int)
    meta_data["fa_pupil_h"] = ((meta_data["fa"] == 1) & (meta_data["pupil_h"] == 1)).astype(int)
    meta_data["fa_pupil_l"] = ((meta_data["fa"] == 1) & (meta_data["pupil_l"] == 1)).astype(int)
    meta_data["miss_rt_h"] = ((meta_data["miss"] == 1) & (meta_data["rt_h"] == 1)).astype(int)
    meta_data["miss_rt_l"] = ((meta_data["miss"] == 1) & (meta_data["rt_l"] == 1)).astype(int)
    meta_data["miss_pupil_h"] = ((meta_data["miss"] == 1) & (meta_data["pupil_h"] == 1)).astype(int)
    meta_data["miss_pupil_l"] = ((meta_data["miss"] == 1) & (meta_data["pupil_l"] == 1)).astype(int)
    meta_data["cr_rt_h"] = ((meta_data["cr"] == 1) & (meta_data["rt_h"] == 1)).astype(int)
    meta_data["cr_rt_l"] = ((meta_data["cr"] == 1) & (meta_data["rt_l"] == 1)).astype(int)
    meta_data["cr_pupil_h"] = ((meta_data["cr"] == 1) & (meta_data["pupil_h"] == 1)).astype(int)
    meta_data["cr_pupil_l"] = ((meta_data["cr"] == 1) & (meta_data["pupil_l"] == 1)).astype(int)

    meta_data["noise_fast_left"] = ((meta_data["left"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_right"] = ((meta_data["right"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_hit"] = ((meta_data["hit"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_fa"] = ((meta_data["fa"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_miss"] = ((meta_data["miss"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_cr"] = ((meta_data["cr"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_conf_h"] = ((meta_data["conf_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_conf_l"] = ((meta_data["conf_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_pupil_h"] = ((meta_data["pupil_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_pupil_l"] = ((meta_data["pupil_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_rt_h"] = ((meta_data["rt_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_rt_l"] = ((meta_data["rt_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_hit_rt_h"] = ((meta_data["hit"] == 1) & (meta_data["rt_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_hit_rt_l"] = ((meta_data["hit"] == 1) & (meta_data["rt_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_hit_pupil_h"] = ((meta_data["hit"] == 1) & (meta_data["pupil_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_hit_pupil_l"] = ((meta_data["hit"] == 1) & (meta_data["pupil_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_fa_rt_h"] = ((meta_data["fa"] == 1) & (meta_data["rt_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_fa_rt_l"] = ((meta_data["fa"] == 1) & (meta_data["rt_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_fa_pupil_h"] = ((meta_data["fa"] == 1) & (meta_data["pupil_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_fa_pupil_l"] = ((meta_data["fa"] == 1) & (meta_data["pupil_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_miss_rt_h"] = ((meta_data["miss"] == 1) & (meta_data["rt_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_miss_rt_l"] = ((meta_data["miss"] == 1) & (meta_data["rt_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_miss_pupil_h"] = ((meta_data["miss"] == 1) & (meta_data["pupil_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_miss_pupil_l"] = ((meta_data["miss"] == 1) & (meta_data["pupil_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_cr_rt_h"] = ((meta_data["cr"] == 1) & (meta_data["rt_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_cr_rt_l"] = ((meta_data["cr"] == 1) & (meta_data["rt_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_cr_pupil_h"] = ((meta_data["cr"] == 1) & (meta_data["pupil_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_cr_pupil_l"] = ((meta_data["cr"] == 1) & (meta_data["pupil_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)

    meta_data["noise_slow_left"] = ((meta_data["left"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_right"] = ((meta_data["right"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_hit"] = ((meta_data["hit"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_fa"] = ((meta_data["fa"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_miss"] = ((meta_data["miss"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_cr"] = ((meta_data["cr"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_conf_h"] = ((meta_data["conf_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_conf_l"] = ((meta_data["conf_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_pupil_h"] = ((meta_data["pupil_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_pupil_l"] = ((meta_data["pupil_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_rt_h"] = ((meta_data["rt_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_rt_l"] = ((meta_data["rt_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_hit_rt_h"] = ((meta_data["hit"] == 1) & (meta_data["rt_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_hit_rt_l"] = ((meta_data["hit"] == 1) & (meta_data["rt_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_hit_pupil_h"] = ((meta_data["hit"] == 1) & (meta_data["pupil_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_hit_pupil_l"] = ((meta_data["hit"] == 1) & (meta_data["pupil_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_fa_rt_h"] = ((meta_data["fa"] == 1) & (meta_data["rt_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_fa_rt_l"] = ((meta_data["fa"] == 1) & (meta_data["rt_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_fa_pupil_h"] = ((meta_data["fa"] == 1) & (meta_data["pupil_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_fa_pupil_l"] = ((meta_data["fa"] == 1) & (meta_data["pupil_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_miss_rt_h"] = ((meta_data["miss"] == 1) & (meta_data["rt_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_miss_rt_l"] = ((meta_data["miss"] == 1) & (meta_data["rt_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_miss_pupil_h"] = ((meta_data["miss"] == 1) & (meta_data["pupil_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_miss_pupil_l"] = ((meta_data["miss"] == 1) & (meta_data["pupil_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_cr_rt_h"] = ((meta_data["cr"] == 1) & (meta_data["rt_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_cr_rt_l"] = ((meta_data["cr"] == 1) & (meta_data["rt_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_cr_pupil_h"] = ((meta_data["cr"] == 1) & (meta_data["pupil_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_cr_pupil_l"] = ((meta_data["cr"] == 1) & (meta_data["pupil_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)

    return meta_data

def compute_contrasts(subj, sessions, contrasts, hemis, baseline_time=(-0.25, -0.15), n_jobs=12):

    tfrs_stim = []
    tfrs_resp = []
    meta_datas = []
    for session in sessions:
        
        print('computing contrasts for {} session {}'.format(subj, session))

        # flip contrast
        if session == 'B':
            hemiss = []
            for i, hemi in enumerate(hemis):
                if hemi=='avg':
                    hemiss.append(hemi)
                elif 'hand' in list(contrasts.keys())[i]:
                    hemiss.append(hemi)
                else:
                    hemiss.append('lh_is_ipsi')
            hemis = hemiss.copy()
        
        with contrast_tfr.Cache() as cache:
            for timelock in ['stimlock', 'resplock']:
                
                data_globstring = os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_*-source.hdf".format(subj, session, timelock,))
                base_globstring = os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_*-source.hdf".format(subj, session, 'stimlock',))

                # tfrs = []
                # tfr_data_filenames = glob.glob(data_globstring)
                # for f in tfr_data_filenames:
                #     tfr = pd.read_hdf(f)
                #     tfr = pd.pivot_table(tfr.reset_index(), values=tfr.columns, index=[
                #                  'trial', 'est_val'], columns='time').stack(-2)
                #     tfr.index.names = ['trial', 'freq', 'area']
                #     tfrs.append(tfr)
                # tfr = pd.concat(tfrs)

                # tfr_data_condition = tfr.groupby(['freq', 'area']).mean()
                # baseline = contrast_tfr.baseline_per_sensor_get(
                #     tfr_data_condition, baseline_time=(-0.3,-0.2))

                # tfr_data_condition = contrast_tfr.baseline_per_sensor_apply(
                #     tfr, baseline=baseline).groupby(['freq', 'area']).mean()
                # X = tfr_data_condition.loc[tfr_data_condition.index.get_level_values('area')=='rh.wang2015atlas.V1d-rh'].groupby(['freq']).mean()

                # from matplotlib.colors import LinearSegmentedColormap
                # cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)
                
                # # variables:
                # times = np.array(X.columns, dtype=float)
                # freqs = np.array(np.unique(X.index.get_level_values('freq')), dtype=float)
                # time_ind = (times>=-0.1) & (times<=1.1)
                
                # # grand average plot:
                # plt.pcolormesh(times[time_ind], freqs, np.array(X)[:,time_ind], vmin=vmin, vmax=vmax, cmap=cmap)
                
                meta_data = load_meta_data(subj, session, timelock, data_folder)
                tfr = contrast_tfr.compute_contrast(contrasts=contrasts, 
                                                    hemis=hemis, 
                                                    data_globstring=data_globstring, 
                                                    base_globstring=base_globstring,
                                                    meta_data=meta_data, 
                                                    baseline_time=baseline_time, 
                                                    baseline_per_condition=False,
                                                    n_jobs=n_jobs, 
                                                    cache=cache)
                
                tfr['subj'] = subj
                tfr['session'] = session
                tfr = tfr.set_index(['cluster', 'subj', 'session', 'contrast', 'hemi'], append=True, inplace=False)
                tfr = tfr.reorder_levels(['subj', 'session', 'cluster', 'contrast', 'hemi', 'freq'])
                if timelock == 'stimlock':
                    meta_datas.append(meta_data)
                    tfrs_stim.append(tfr)
                elif timelock == 'resplock':
                    tfrs_resp.append(tfr)
    
    # fix session inbalance:
    nr_sessions = len(meta_datas)
    if nr_sessions > 1:
        for c in contrasts.keys():
            trial_types = contrasts[c][0]
            if len(trial_types) == 1:
                fractions = np.array([sum(meta_datas[s][trial_types[0]]==1) for s in range(nr_sessions)])
                fractions = fractions / sum(fractions)
            else:
                fractions = np.array([[sum(meta_datas[s][trial_types[i]]==1) for i in range(len(trial_types))] for s in range(nr_sessions)])
                fractions = fractions.min(axis=1) / sum(fractions.min(axis=1))
            for s in range(nr_sessions):
                ind = tfrs_stim[s].index.get_level_values('contrast') == c
                tfrs_stim[s].loc[ind,:] = tfrs_stim[s].loc[ind,:] * fractions[s]
                ind = tfrs_resp[s].index.get_level_values('contrast') == c
                tfrs_resp[s].loc[ind,:] = tfrs_resp[s].loc[ind,:] * fractions[s]

    # concat sessions:
    tfrs_stim = pd.concat(tfrs_stim)
    tfrs_resp = pd.concat(tfrs_resp)

    # sum across sessions:
    tfrs_stim = tfrs_stim.groupby(['subj', 'cluster', 'contrast', 'freq']).sum()
    tfrs_resp = tfrs_resp.groupby(['subj', 'cluster', 'contrast', 'freq']).sum()

    # save:
    tfrs_stim.to_hdf(os.path.join(data_folder, 'source_level', 'contrasts', 'tfr_contrasts_stimlock_{}.hdf'.format(subj)), 'tfr')
    tfrs_resp.to_hdf(os.path.join(data_folder, 'source_level', 'contrasts', 'tfr_contrasts_resplock_{}.hdf'.format(subj)), 'tfr')

def load_contrasts(subj, timelock):
    tfr = pd.read_hdf(os.path.join(data_folder, 'source_level', 'contrasts', 'tfr_contrasts_{}_{}.hdf'.format(timelock, subj)))
    return tfr

def plot_tfr(tfr, time_cutoff, vmin, vmax, tl, cluster_correct=False, threshold=0.05, plot_colorbar=False, ax=None):

    # colorbar:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)
    
    # variables:
    times = np.array(tfr.columns, dtype=float)
    freqs = np.array(np.unique(tfr.index.get_level_values('freq')), dtype=float)
    time_ind = (times>time_cutoff[0]) & (times<time_cutoff[1])
    time_ind = (times>time_cutoff[0]) & (times<time_cutoff[1])

    # data:
    X = np.stack([tfr.loc[tfr.index.isin([subj], level='subj'), time_ind].values for subj in np.unique(tfr.index.get_level_values('subj'))])

    # grand average plot:
    cax = ax.pcolormesh(times[time_ind], freqs, X.mean(axis=0), vmin=vmin, vmax=vmax, cmap=cmap)
        
    # cluster stats:
    if cluster_correct:
        if tl  == 'stimlock':
            test_data = X[:,:,times[time_ind]>0]
            times_test_data = times[time_ind][times[time_ind]>0]
        else:
            test_data = X.copy()
            times_test_data = times[time_ind]
        try:
            T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(test_data, threshold={'start':0, 'step':0.2}, 
                                                                                        connectivity=None, tail=0, n_permutations=1000, n_jobs=24)
            sig = cluster_p_values.reshape((test_data.shape[1], test_data.shape[2]))
            ax.contour(times_test_data, freqs, sig, (threshold,), linewidths=0.5, colors=('black'))
        except:
            pass
    ax.axvline(0, ls='--', lw=0.75, color='black',)
    if plot_colorbar:
        plt.colorbar(cax, ticks=[vmin, 0, vmax])
    return ax

def plot_tfr_per_atlas(tfrs_stim, tfrs_resp, atlas, cluster_correct=False):

    nr_clusters = len(atlas.keys())

    fig = plt.figure(figsize=(3,nr_clusters*1.25)) 

    ratio = (0.2--0.6) / (0.6--0.35)

    gs = matplotlib.gridspec.GridSpec(nr_clusters, 2, width_ratios=[1,ratio])
    # gs.update(wspace=0.1, hspace=0.1, top=0.98, bottom=0.03)

    for i, cluster in enumerate(atlas.keys()):
        for j, timelock in enumerate(['stimlock', 'resplock']):
            
            ax = plt.subplot(gs[i,j])

            # tfr to plot:
            if timelock == 'stimlock':
                tfr = tfrs_stim.loc[tfrs_stim.index.isin([cluster], level='cluster') & 
                                tfrs_stim.index.isin([contrast_name], level='contrast')]
            elif timelock == 'resplock':
                tfr = tfrs_resp.loc[tfrs_resp.index.isin([cluster], level='cluster') & 
                                tfrs_resp.index.isin([contrast_name], level='contrast')]

            # time:
            if timelock  == 'stimlock':
                time_cutoff = (-0.35, 0.6)
                xlabel = 'Time from stimulus (s)'
            if timelock  == 'resplock':
                time_cutoff = (-0.6, 0.2)
                xlabel = 'Time from report (s)'

            # vmin vmax:
            if contrast_name == 'all':
                vmin, vmax = (-15, 15)
            else:
                vmin, vmax = (-10, 10)

            plot_tfr(tfr, time_cutoff, vmin, vmax, timelock, 
                        cluster_correct=cluster_correct, threshold=0.05, ax=ax)
            if i == 0:
                if timelock == 'stimlock':
                    ax.set_title('{} contrast (N={})'.format(contrast_name, len(subjects)))
            if i == (nr_clusters-1):
                ax.set_xlabel(xlabel)
            if timelock == 'stimlock':
                ax.text(0.5, 0.85, cluster, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes,
                    size=7)
                ax.set_ylabel('Frequency (Hz)')
                ax.axvline(-0.25, ls=':', lw=0.75, color='black',)
                ax.axvline(-0.15, ls=':', lw=0.75, color='black',)
            elif timelock == 'resplock':
                ax.tick_params(left=False, labelleft='off')
    #fig.tight_layout()
    gs.tight_layout(fig)
    gs.update(wspace=0.05, hspace=0.05)

    return fig

def plot_tfr_selected_rois(contrast_name, tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, cluster_names_row1, cluster_names_row2, cluster_correct=False):

    nr_clusters = len(clusters_row1)

    fig = plt.figure(figsize=(nr_clusters*0.9, 2.5)) 

    ratio = (0.1--0.4) / (1.1--0.3)

    gs = matplotlib.gridspec.GridSpec(2, nr_clusters*2, width_ratios=list(np.tile([1,ratio], nr_clusters)))
    # gs.update(wspace=0.1, hspace=0.1, top=0.98, bottom=0.03)

    for i in range(2):
        for j, timelock in enumerate(list(np.tile(['stimlock', 'resplock'], nr_clusters))):
            
            ax = plt.subplot(gs[i,j])
            
            # select cluster:
            if i == 0:
                cluster = clusters_row1[int(j/2)]
                cluster_name = cluster_names_row1[int(j/2)]
            else:
                cluster = clusters_row2[int(j/2)]
                cluster_name = cluster_names_row2[int(j/2)]


            # tfr to plot:
            if timelock == 'stimlock':
                tfr = tfrs_stim.loc[tfrs_stim.index.isin([cluster], level='cluster') & 
                                tfrs_stim.index.isin([contrast_name], level='contrast')]
            elif timelock == 'resplock':
                tfr = tfrs_resp.loc[tfrs_resp.index.isin([cluster], level='cluster') & 
                                tfrs_resp.index.isin([contrast_name], level='contrast')]

            # time:
            if timelock  == 'stimlock':
                time_cutoff = (-0.3, 1.1)
                xlabel = 'Time from stimulus (s)'
            if timelock  == 'resplock':
                time_cutoff = (-0.4, 0.1)
                xlabel = 'Time from report (s)'
            
            # vmin vmax:
            if contrast_name == 'all':
                vmin, vmax = (-15, 15)
            else:
                vmin, vmax = (-10, 10)

            if j == 3:
                plot_colorbar = False
            else:
                plot_colorbar = False

            plot_tfr(tfr, time_cutoff, vmin, vmax, timelock, 
                        cluster_correct=cluster_correct, threshold=0.05, 
                        plot_colorbar=plot_colorbar, ax=ax)
        
            if timelock == 'stimlock':
                ax.text(0.5, 0.85, cluster_name, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes,
                    size=6)
                ax.axvline(-0.25, ls=':', lw=0.75, color='black',)
                ax.axvline(-0.15, ls=':', lw=0.75, color='black',)

            if (clusters_row1[int(j/2)] in ['vfcPrimary', 'vfcEarly', 'JWG_M1']) & (i>0):
               ax.axis('off') 

            # take care of tickmarks:
            if i == 0:
                ax.get_xaxis().set_ticks([])
            if not j == 0:
                ax.get_yaxis().set_ticks([])
                #ax.tick_params(left=False, labelleft='off')
            
            # labels:
            if i == 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                if i == 0:
                    ax.set_title('{} contrast (N={})'.format(contrast_name, len(subjects)))
                ax.set_ylabel('Frequency (Hz)')

            if (clusters_row1[int(j/2)] in ['vfcPrimary', 'vfcEarly', 'JWG_M1']) & (i>0):
                for artist in plt.gca().lines + plt.gca().collections:
                    artist.remove()

    #fig.tight_layout()
    gs.tight_layout(fig)
    gs.update(wspace=0.05, hspace=0.05)

    return fig

def plot_timecourses_selected_rois(contrast_names, tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, cluster_names_row1, cluster_names_row2, freqs=(12,36)):

    nr_clusters = len(clusters_row1)

    fig = plt.figure(figsize=(nr_clusters*0.9, 2.5)) 

    ratio = (0.1--0.4) / (1.1--0.3)

    gs = matplotlib.gridspec.GridSpec(2, nr_clusters*2, width_ratios=list(np.tile([1,ratio], nr_clusters)))
    # gs.update(wspace=0.1, hspace=0.1, top=0.98, bottom=0.03)

    for i in range(2):
        for j, timelock in enumerate(list(np.tile(['stimlock', 'resplock'], nr_clusters))):
            
            ax = plt.subplot(gs[i,j])
            # select cluster:
            if i == 0:
                cluster = clusters_row1[int(j/2)]
                cluster_name = cluster_names_row1[int(j/2)]
            else:
                cluster = clusters_row2[int(j/2)]
                cluster_name = cluster_names_row2[int(j/2)]

            
                
            for c in contrast_names:

                # tfr to plot:
                if timelock == 'stimlock':
                    tfr = tfrs_stim.loc[tfrs_stim.index.isin([cluster], level='cluster') & 
                                    tfrs_stim.index.isin([c], level='contrast')]
                elif timelock == 'resplock':
                    tfr = tfrs_resp.loc[tfrs_resp.index.isin([cluster], level='cluster') & 
                                    tfrs_resp.index.isin([c], level='contrast')]

                # time:
                if timelock  == 'stimlock':
                    time_cutoff = (-0.3, 1.1)
                    xlabel = 'Time from stimulus (s)'
                if timelock  == 'resplock':
                    time_cutoff = (-0.4, 0.1)
                    xlabel = 'Time from report (s)'

                # vmin vmax:
                if c == 'all':
                    vmin, vmax = (-15, 15)
                elif c == 'hand':
                    if freqs[0] < 40:
                        vmin, vmax = (-25, 5)
                    else:
                        vmin, vmax = (-5, 20)
                else:
                    vmin, vmax = (-20, 20)

                if j == 3:
                    plot_colorbar = False
                else:
                    plot_colorbar = False
                
                if ('hit' in c) or ('fa' in c):
                    color = 'orange'
                else:
                    color = 'forestgreen'
                if ('hit' in c) or ('cr' in c):
                    ls = '-'
                else:
                    ls = '--'

                # collapse across freq:
                tfr = tfr.loc[(tfr.index.get_level_values('freq')>=freqs[0]) & (tfr.index.get_level_values('freq')<=freqs[1]),:].groupby(['subj']).mean()
                
                # remove some time:
                time_ind = (tfr.columns >= time_cutoff[0]) & (tfr.columns <= time_cutoff[1])
                tfr = tfr.loc[:,time_ind]
                
                times = np.array(tfr.columns, dtype=float)
                ax.fill_between(times, tfr.mean(axis=0)-tfr.sem(axis=0), tfr.mean(axis=0)+tfr.sem(axis=0), alpha=0.2, color=color)
                ax.plot(times, tfr.mean(axis=0), color=color, ls=ls, lw=0.75)
                
                ax.set_ylim(vmin, vmax)

                if timelock == 'stimlock':
                    ax.text(0.5, 0.85, cluster_name, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,
                        size=6)
                    # ax.axvspan(xmin=-0.25, xmax=-0.15, color='black', alpha=0.2)
                ax.axvline(0, lw=0.5, color='black', alpha=0.25)

            # take care of tickmarks:
            if i == 0:
                ax.get_xaxis().set_ticks([])
            if not j == 0:
                ax.get_yaxis().set_ticks([])
                #ax.tick_params(left=False, labelleft='off')
                        
            # labels:
            if i == 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                if i == 0:
                    ax.set_title('{} contrast (N={})'.format("SDT", len(subjects)))
                ax.set_ylabel('Power (% signal change)')

            if (clusters_row1[int(j/2)] in ['vfcPrimary', 'vfcEarly', 'HCPMMP1_premotor', 'JWG_M1']) & (i>0):
                for artist in plt.gca().lines + plt.gca().collections:
                    artist.remove()

    #fig.tight_layout()
    gs.tight_layout(fig)
    gs.update(wspace=0.05, hspace=0.05)

    return fig

def get_cluster_vertices(clusters, hemi, subj='fsaverage'):
    
    from pymeg import source_reconstruction as sr

    # load labels:
    labels = sr.get_labels(subject=subj, filters=['*wang*.label', '*JWG*.label'], annotations=['HCPMMP1'] )
    labels = sr.labels_exclude(labels=labels, exclude_filters=['wang2015atlas.IPS4', 'wang2015atlas.IPS5', 
                                                    'wang2015atlas.SPL', 'JWG_lat_Unknown'])
    labels = sr.labels_remove_overlap(labels=labels, priority_filters=['wang', 'JWG'])

    # globals().update(locals())

    # obtain vertices:
    vertices = {}
    for cluster in clusters.keys():
        rois = clusters[cluster]
        # vertices_cluster = np.concatenate([l.get_vertices_used() for l in labels for roi in rois if (l.hemi==hemi)&(l.name==roi)])
        vertices_cluster = []
        for roi in rois:
            for l in labels:
                if (l.hemi==hemi)&(l.name==roi):
                    vertices_cluster.append(l.get_vertices_used())
        print(vertices_cluster)
        try:
            vertices_cluster = np.concatenate(vertices_cluster)
        except:
            print(roi)
            shell()
        vertices[cluster] = vertices_cluster

    return vertices

def plot_surface_clusters(clusters, hemi, subj='fsaverage'):
    
    from nilearn import datasets, plotting

    # obtain vertices:
    vertices = get_cluster_vertices(clusters=clusters, hemi=hemi, subj=subj,)

    # load surface:
    fsaverage = datasets.fetch_surf_fsaverage5()

    # create map to plot:
    to_plot = np.zeros(10242)
    this_cluster = 1
    for cluster in vertices.keys():
        to_plot[vertices[cluster]] = this_cluster
        this_cluster += 1

    # plot:
    for i, view in enumerate(['medial', 'lateral', 'posterior', 'anterior']):
        print(view)
        fig = plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=to_plot,
                        hemi='left', view=view,
                        bg_map=fsaverage['sulc_left'], bg_on_data=True,
                        darkness=.5)
        fig.savefig('/media/external3/JW/rois_{}.{}'.format(view))

def plot_surface_contrast(tfrs, contrast_name, freq_cutoffs, time_cutoffs, vmax, clusters, subj='fsaverage'):
    
    from nilearn import datasets, plotting

    fig, axes = plt.subplots(subplot_kw={'projection': '3d'}, nrows=2, ncols=4, **{'figsize':(8,4)})
    for column, hemi in enumerate(['rh', 'lh']):
                
        # obtain vertices:
        vertices = get_cluster_vertices(clusters=clusters, hemi=hemi, subj=subj,)

        # load surface:
        fsaverage = datasets.fetch_surf_fsaverage5()

        # subselect contrast:
        tfr = tfrs.loc[tfrs.index.isin([contrast_name], level='contrast')]

        # collapse across freq:
        tfr = tfr.loc[(tfr.index.get_level_values('freq')>=freq_cutoffs[0]) & \
                        (tfr.index.get_level_values('freq')<=freq_cutoffs[1]),:].groupby(['cluster']).mean()
        
        # collapse across time:
        tfr = tfr.loc[:,(tfr.columns>=time_cutoffs[0]) & (tfr.columns<=time_cutoffs[1])].mean(axis=1)

        # create map to plot:
        to_plot = np.zeros(10242)
        for cluster in vertices.keys():
            to_plot[vertices[cluster]] = tfr.loc[tfr.index.get_level_values('cluster')==cluster]

        # colorbar:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)

        # plot:
        if hemi == 'lh':
            for i, view in enumerate(['lateral', 'medial', 'posterior', 'anterior']):
                plotting.plot_surf_stat_map(surf_mesh=fsaverage['pial_left'], stat_map=to_plot, bg_map=fsaverage['sulc_left'],
                                            hemi='left', view=view, cmap=cmap, symmetric_cbar=True, vmax=vmax, colorbar=True, 
                                            bg_on_data=True, axes=axes[column][i],)
        elif hemi == 'rh':
            for i, view in enumerate(['medial', 'lateral', 'posterior', 'anterior']):
                plotting.plot_surf_stat_map(surf_mesh=fsaverage['pial_right'], stat_map=to_plot, bg_map=fsaverage['sulc_right'],
                                            hemi='left', view=view, cmap=cmap, symmetric_cbar=True, vmax=vmax, colorbar=True, 
                                            bg_on_data=True, axes=axes[column][i],)
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    
    compute = False
    plot = True
    cluster_correct = False

    # subjects:
    subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    # subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13',]
    # subjects = ['jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    
    # exclude subjects:
    subjects = select_subjects(subjects, ["noise_fast_fa_pupil_l", "noise_fast_fa_pupil_h", 
                                          "noise_fast_miss_pupil_l", "noise_fast_miss_pupil_h",
                                          "noise_slow_fa_pupil_l", "noise_slow_fa_pupil_h",
                                          "noise_slow_miss_pupil_l", "noise_slow_miss_pupil_h",], cutoff=2)
    
    # subjects = ['jw01',
    #             'jw02',
    #             'jw08',
    #             'jw10',
    #             'jw14',
    #             'jw15',
    #             'jw18',
    #             'jw19',
    #             'jw20',
    #             'jw21',
    #             'jw22',
    #             'jw30'
    #             ]

    # get clusters:
    all_clusters, visual_field_clusters, glasser_clusters, jwg_clusters = atlas_glasser.get_clusters()

    # define contrasts:
    contrasts = {
    # 'all': (['all'], [1]),
    # 'choice': (['hit', 'fa', 'miss', 'cr'], (1, 1, -1, -1)),
    # 'stimulus': (['hit', 'fa', 'miss', 'cr'], (1, -1, 1, -1)),
    # 'hand': (['left', 'right'], (1, -1)),
    # 'confidence': (['conf_h', 'conf_l'], (1, -1)),
    # 'noise': (['noise_fast', 'noise_slow'], (1, -1)),
    # 'pupil': (['pupil_h', 'pupil_l'], (1, -1)),
    # 'rt': (['rt_l', 'rt_h'], (1, -1)),
    # 'hit': (['hit'], [1]),
    # 'fa': (['fa'], [1]),
    # 'miss': (['miss'], [1]),
    # 'cr': (['cr'], [1]),
    # "hit_rt_h": (['hit_rt_h'], [1]),
    # "hit_rt_l": (['hit_rt_l'], [1]),
    # "hit_pupil_h": (['hit_pupil_h'], [1]),
    # "hit_pupil_l": (['hit_pupil_l'], [1]),
    # "fa_rt_h": (['fa_rt_h'], [1]),
    # "fa_rt_l": (['fa_rt_l'], [1]),
    # "fa_pupil_h": (['fa_pupil_h'], [1]),
    # "fa_pupil_l": (['fa_pupil_l'], [1]),
    # "miss_rt_h": (['miss_rt_h'], [1]),
    # "miss_rt_l": (['miss_rt_l'], [1]),
    # "miss_pupil_h": (['miss_pupil_h'], [1]),
    # "miss_pupil_l": (['miss_pupil_l'], [1]),
    # "cr_rt_h": (['cr_rt_h'], [1]),
    # "cr_rt_l": (['cr_rt_l'], [1]),
    # "cr_pupil_h": (['cr_pupil_h'], [1]),
    # "cr_pupil_l": (['cr_pupil_l'], [1]),

    'noise_fast': (['noise_fast'], [1]),
    'noise_fast_choice': (['noise_fast_hit', 'noise_fast_fa', 'noise_fast_miss', 'noise_fast_cr'], (1, 1, -1, -1)),
    'noise_fast_stimulus': (['noise_fast_hit', 'noise_fast_fa', 'noise_fast_miss', 'noise_fast_cr'], (1, -1, 1, -1)),
    'noise_fast_hand': (['noise_fast_left', 'noise_fast_right'], (1, -1)),
    'noise_fast_confidence': (['noise_fast_conf_h', 'noise_fast_conf_l'], (1, -1)),
    'noise_fast_pupil': (['noise_fast_pupil_h', 'noise_fast_pupil_l'], (1, -1)),
    'noise_fast_rt': (['noise_fast_rt_l', 'noise_fast_rt_h'], (1, -1)),
    'noise_fast_hit': (['noise_fast_hit'], [1]),
    'noise_fast_fa': (['noise_fast_fa'], [1]),
    'noise_fast_miss': (['noise_fast_miss'], [1]),
    'noise_fast_cr': (['noise_fast_cr'], [1]),
    "noise_fast_hit_rt_h": (['noise_fast_hit_rt_h'], [1]),
    "noise_fast_hit_rt_l": (['noise_fast_hit_rt_l'], [1]),
    "noise_fast_hit_pupil_h": (['noise_fast_hit_pupil_h'], [1]),
    "noise_fast_hit_pupil_l": (['noise_fast_hit_pupil_l'], [1]),
    "noise_fast_fa_rt_h": (['noise_fast_fa_rt_h'], [1]),
    "noise_fast_fa_rt_l": (['noise_fast_fa_rt_l'], [1]),
    "noise_fast_fa_pupil_h": (['noise_fast_fa_pupil_h'], [1]),
    "noise_fast_fa_pupil_l": (['noise_fast_fa_pupil_l'], [1]),
    "noise_fast_miss_rt_h": (['noise_fast_miss_rt_h'], [1]),
    "noise_fast_miss_rt_l": (['noise_fast_miss_rt_l'], [1]),
    "noise_fast_miss_pupil_h": (['noise_fast_miss_pupil_h'], [1]),
    "noise_fast_miss_pupil_l": (['noise_fast_miss_pupil_l'], [1]),
    "noise_fast_cr_rt_h": (['noise_fast_cr_rt_h'], [1]),
    "noise_fast_cr_rt_l": (['noise_fast_cr_rt_l'], [1]),
    "noise_fast_cr_pupil_h": (['noise_fast_cr_pupil_h'], [1]),
    "noise_fast_cr_pupil_l": (['noise_fast_cr_pupil_l'], [1]),

    'noise_slow': (['noise_slow'], [1]),
    'noise_slow_choice': (['noise_slow_hit', 'noise_slow_fa', 'noise_slow_miss', 'noise_slow_cr'], (1, 1, -1, -1)),
    'noise_slow_stimulus': (['noise_slow_hit', 'noise_slow_fa', 'noise_slow_miss', 'noise_slow_cr'], (1, -1, 1, -1)),
    'noise_slow_hand': (['noise_slow_left', 'noise_slow_right'], (1, -1)),
    'noise_slow_confidence': (['noise_slow_conf_h', 'noise_slow_conf_l'], (1, -1)),
    'noise_slow_pupil': (['noise_slow_pupil_h', 'noise_slow_pupil_l'], (1, -1)),
    'noise_slow_rt': (['noise_slow_rt_l', 'noise_slow_rt_h'], (1, -1)),
    'noise_slow_hit': (['noise_slow_hit'], [1]),
    'noise_slow_fa': (['noise_slow_fa'], [1]),
    'noise_slow_miss': (['noise_slow_miss'], [1]),
    'noise_slow_cr': (['noise_slow_cr'], [1]),
    "noise_slow_hit_rt_h": (['noise_slow_hit_rt_h'], [1]),
    "noise_slow_hit_rt_l": (['noise_slow_hit_rt_l'], [1]),
    "noise_slow_hit_pupil_h": (['noise_slow_hit_pupil_h'], [1]),
    "noise_slow_hit_pupil_l": (['noise_slow_hit_pupil_l'], [1]),
    "noise_slow_fa_rt_h": (['noise_slow_fa_rt_h'], [1]),
    "noise_slow_fa_rt_l": (['noise_slow_fa_rt_l'], [1]),
    "noise_slow_fa_pupil_h": (['noise_slow_fa_pupil_h'], [1]),
    "noise_slow_fa_pupil_l": (['noise_slow_fa_pupil_l'], [1]),
    "noise_slow_miss_rt_h": (['noise_slow_miss_rt_h'], [1]),
    "noise_slow_miss_rt_l": (['noise_slow_miss_rt_l'], [1]),
    "noise_slow_miss_pupil_h": (['noise_slow_miss_pupil_h'], [1]),
    "noise_slow_miss_pupil_l": (['noise_slow_miss_pupil_l'], [1]),
    "noise_slow_cr_rt_h": (['noise_slow_cr_rt_h'], [1]),
    "noise_slow_cr_rt_l": (['noise_slow_cr_rt_l'], [1]),
    "noise_slow_cr_pupil_h": (['noise_slow_cr_pupil_h'], [1]),
    "noise_slow_cr_pupil_l": (['noise_slow_cr_pupil_l'], [1]),
    }

    # hemis:
    hemis = [
            # 'avg', 'avg', 'avg', 'rh_is_ipsi', 'avg', 'avg', 'avg', 'avg', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi',
            'avg', 'avg', 'avg', 'rh_is_ipsi', 'avg', 'avg', 'avg', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi',
            'avg', 'avg', 'avg', 'rh_is_ipsi', 'avg', 'avg', 'avg', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi', 'rh_is_ipsi',
            ]

    contrasts_to_plot = {
    'noise_fast': (['noise_slow'], [1]),
    'noise_fast_choice': (['noise_fast_hit', 'noise_fast_fa', 'noise_fast_miss', 'noise_fast_cr'], (1, 1, -1, -1)),
    'noise_fast_stimulus': (['noise_fast_hit', 'noise_fast_fa', 'noise_fast_miss', 'noise_fast_cr'], (1, -1, 1, -1)),
    'noise_fast_hand': (['noise_fast_left', 'noise_fast_right'], (1, -1)),
    'noise_fast_confidence': (['noise_fast_conf_h', 'noise_fast_conf_l'], (1, -1)),
    'noise_fast_pupil': (['noise_fast_pupil_h', 'noise_fast_pupil_l'], (1, -1)),
    'noise_fast_rt': (['noise_fast_rt_l', 'noise_fast_rt_h'], (1, -1)),

    'noise_slow': (['noise_slow'], [1]),
    'noise_slow_choice': (['noise_slow_hit', 'noise_slow_fa', 'noise_slow_miss', 'noise_slow_cr'], (1, 1, -1, -1)),
    'noise_slow_stimulus': (['noise_slow_hit', 'noise_slow_fa', 'noise_slow_miss', 'noise_slow_cr'], (1, -1, 1, -1)),
    'noise_slow_hand': (['noise_slow_left', 'noise_slow_right'], (1, -1)),
    'noise_slow_confidence': (['noise_slow_conf_h', 'noise_slow_conf_l'], (1, -1)),
    'noise_slow_pupil': (['noise_slow_pupil_h', 'noise_slow_pupil_l'], (1, -1)),
    'noise_slow_rt': (['noise_slow_rt_l', 'noise_slow_rt_h'], (1, -1)),
    }

    # compute contrasts:
    if compute:
        for subj in subjects:
            if subj == 'jw16':
                sessions = ['B']
            elif subj == 'jw24':
                sessions = ['A']
            elif subj == 'jw30':
                sessions = ['A']
            else:
                sessions = ['A', 'B']
            # sessions = which_sessions[subj]
            compute_contrasts(subj, sessions, contrasts, hemis, n_jobs=16)

    # plot:
    if plot:

        file_format = 'png'

        tfrs_stim = []
        tfrs_resp = []
        for subj in subjects:
            print(subj)
            tfr_stim = load_contrasts(subj, 'stimlock')
            tfr_resp = load_contrasts(subj, 'resplock')
            tfrs_stim.append(tfr_stim)
            tfrs_resp.append(tfr_resp)
        tfrs_stim = pd.concat(tfrs_stim)
        tfrs_resp = pd.concat(tfrs_resp)

        # clusters:
        clusters_row1 = ['vfcPrimary', 'vfcEarly', 'vfcV3ab', 'vfcIPS01', 'vfcIPS23', 'JWG_IPS_PCeS', 'HCPMMP1_dlpfc', 'vfcFEF', 'HCPMMP1_premotor', 'JWG_M1']
        clusters_row2 = ['vfcPrimary', 'vfcEarly', 'vfcVO', 'vfcPHC', 'vfcTO', 'vfcLO', 'post_medial_frontal', 'vent_medial_frontal', 'ant_medial_frontal', 'JWG_M1']
        cluster_names_row1 = ['V1', 'V2-V4', 'V3A/B', 'IPS0/1', 'IPS2/3', 'IPS/PostCeS', 'dlPFC', 'FEF', 'Premotor', 'M1']
        cluster_names_row2 = ['V1', 'V2-V4', 'Vent. Occ', 'PHC', 'MT+', 'Lat. Occ', 'pmPFC', 'vmPFC', 'amPFC', 'M1']

        for contrast_name in contrasts_to_plot.keys():
            
            ########################
            # TFR SELECTED REGIONS #
            ########################

            fig = plot_tfr_selected_rois(contrast_name, tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, 
                                            cluster_names_row1, cluster_names_row2, cluster_correct=cluster_correct)
            fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}.{}'.format(contrast_name, 'selection', file_format)), dpi=300)

            fig = plot_timecourses_selected_rois([contrast_name], tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, 
                                            cluster_names_row1, cluster_names_row2, freqs=(12,36))
            fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_timecourse_low.{}'.format(contrast_name, 'selection', file_format)), dpi=300)

            fig = plot_timecourses_selected_rois([contrast_name], tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, 
                                            cluster_names_row1, cluster_names_row2, freqs=(50,150))
            fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_timecourse_high.{}'.format(contrast_name, 'selection', file_format)), dpi=300)

        for add in ['', 'noise_slow_', 'noise_fast_',]:
        
            fig = plot_timecourses_selected_rois(['{}hit'.format(add), '{}fa'.format(add), '{}miss'.format(add), '{}cr'.format(add)], 
                                            tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, 
                                            cluster_names_row1, cluster_names_row2, freqs=(12,36))
            fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_timecourse_low_{}.{}'.format('sdt_slopes', 'selection', add, file_format)), dpi=300)

            fig = plot_timecourses_selected_rois(['{}hit'.format(add), '{}fa'.format(add), '{}miss'.format(add), '{}cr'.format(add)], 
                                            tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, 
                                            cluster_names_row1, cluster_names_row2, freqs=(50,150))
            fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_timecourse_high_{}.{}'.format('sdt_slopes', 'selection', add, file_format)), dpi=300)

        for add in [('noise_slow', 'pupil_h'), ('noise_slow', 'pupil_l')]:
        
            fig = plot_timecourses_selected_rois(['{}_hit_{}'.format(add[0], add[1]), '{}_fa_{}'.format(add[0], add[1]), '{}_miss_{}'.format(add[0], add[1]), '{}_cr_{}'.format(add[0], add[1])], 
                                            tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, 
                                            cluster_names_row1, cluster_names_row2, freqs=(12,36))
            fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_timecourse_low_{}_{}.{}'.format('sdt_slopes', 'selection', add[0], add[1], file_format)), dpi=300)

            fig = plot_timecourses_selected_rois(['{}_hit_{}'.format(add[0], add[1]), '{}_fa_{}'.format(add[0], add[1]), '{}_miss_{}'.format(add[0], add[1]), '{}_cr_{}'.format(add[0], add[1])], 
                                            tfrs_stim, tfrs_resp, clusters_row1, clusters_row2, 
                                            cluster_names_row1, cluster_names_row2, freqs=(50,150))
            fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_timecourse_high_{}_{}.{}'.format('sdt_slopes', 'selection', add[0], add[1], file_format)), dpi=300)

        # # ########################
        # # #   WHOLE BRAIN MAPS   #
        # # ########################
        
        # for contrast_name in contrasts_to_plot.keys():

        #     # whole brain maps:
        #     if 'hand' in contrast_name:
        #         tfrs = tfrs_resp.copy()
        #         freq_cutoffs=[12,36]
        #         time_cutoffs=[-0.5,0]
        #     else:
        #         tfrs = tfrs_stim.copy()
        #         freq_cutoffs=[50,150]
        #         time_cutoffs=[0.1,0.9]
        #     fig = plot_surface_contrast(tfrs=tfrs, contrast_name=contrast_name, 
        #                                 freq_cutoffs=freq_cutoffs, time_cutoffs=time_cutoffs, 
        #                                 clusters=all_clusters, vmax=10, subj='fsaverage')
        #     fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}.{}'.format(contrast_name, 'whole_brain', file_format)), dpi=300)
            
        #     # ########################
        #     # #     TFR PER ATLAS    #
        #     # ########################

        #     # for atlas, atlas_name in zip((visual_field_clusters, glasser_clusters, jwg_clusters), ('Wang', 'Glasser', 'JW')):
                
        #     #     fig = plot_tfr_per_atlas(tfrs_stim, tfrs_resp, atlas, cluster_correct=cluster_correct)
        #     #     fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}.{}'.format(contrast_name, atlas_name)))