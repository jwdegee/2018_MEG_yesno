import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from joblib import Parallel, delayed

# SET PYMEG_CACHE_DIR=\\discovery1\bcm-neuro-mcginley\JW\cache
# os.system('SET PYMEG_CACHE_DIR=\\\\discovery1\\bcm-neuro-mcginley\\JW\\cache')

from IPython import embed as shell

import pymeg
from pymeg import preprocessing as prep
from pymeg import tfr as tfr


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

#data_folder = '/home/jw/share/data/'
#fig_folder = '/home/jw/share/figures/'
data_folder = '\\\\discovery1\\bcm-neuro-mcginley\\JW\\data'
fig_folder = '\\\\discovery1\\bcm-neuro-mcginley\\JW\\figures'
tfr_params = tfr_params = json.load(open('tfr_params.json'))
channels = list(pd.read_json("channels.json")[0])
foi = list(np.arange(4,162,2))

def plot_by_freq(resp, **kwargs):
    if 'vmin' not in list(kwargs.keys()):
        kwargs['vmin'] = -0.5
    if 'vmax' not in list(kwargs.keys()):
        kwargs['vmax'] = 0.5
    if 'cmap' not in list(kwargs.keys()):
        kwargs['cmap'] = 'RdBu_r'

    n_groups = len(np.unique(resp.index.get_level_values('cgroup')))
    for i, (c, m) in enumerate(resp.groupby(level='cgroup')):
        time = np.unique(m.columns.get_level_values('time'))
        tc = np.array([time[0]] + list([(low + (high - low) / 2.)
                                        for low, high in zip(time[:-1], time[1:])]) + [time[-1]])
        freqs = np.unique(m.index.get_level_values('freq'))
        fc = np.array([freqs[0]] + list([(low + (high - low) / 2.)
                                         for low, high in zip(freqs[:-1], freqs[1:])]) + [freqs[-1]])

        plt.subplot(1, n_groups, i + 1)
        plt.pcolormesh(tc, fc,
                       m.values, **kwargs)
        #plt.xlim([-0.4, 1.1])
        plt.ylim([min(freqs), max(freqs)])
        if i > 0:
            plt.yticks([])
        else:
            plt.ylabel('Frequency')
            plt.xlabel('Time')
        plt.xticks([-.4, 0, .4, .8])

def baseline_get(tfr, baseline=(-0.4, 0)):
    '''
    Get average baseline
    '''
    time = tfr.columns.get_level_values('time').values.astype(float)
    id_base =  (time > baseline[0]) & (time < baseline[1])
    base = tfr.loc[:, id_base].groupby('freq').mean().mean(axis=1)  # This should be len(#Freq)
    return base

def baseline_per_sensor_get(tfr, baseline=(-0.4, 0)):
    '''
    Get average baseline
    '''
    time = tfr.columns.get_level_values('time').values.astype(float)
    id_base =  (time > baseline[0]) & (time < baseline[1])
    base = tfr.loc[:, id_base].groupby(['freq', 'channel']).mean().mean(axis=1)  # This should be len(nr_freqs * nr_hannels)
    return base

def baseline_apply(tfr, baseline):
    '''
    Baseline correction by dividing by average baseline
    '''
    def div(x):
        bval = float(baseline.loc[x.index.get_level_values('freq').values[0]])
        return (x - bval) / bval * 100
    return tfr.groupby(level='freq').apply(div)
    
def baseline_per_sensor_apply(tfr, baseline):
    '''
    Baseline correction by dividing by average baseline
    '''
    def div(x):
        bval = float(baseline.loc[baseline.index.isin([x.index.get_level_values('freq').values[0]], level='freq') & 
                                  baseline.index.isin([x.index.get_level_values('channel').values[0]], level='channel')])
        return (x - bval) / bval * 100
    return tfr.groupby(['freq', 'channel']).apply(div)

def rt_diffs(subjects):
    
    max_rt = []
    min_rt = []
    
    for subj in subjects:
        meta_ses = []
        for session in ["A", "B"]:
            runs = sorted([run.split('/')[-1] for run in glob.glob(os.path.join(data_folder, "raw", subj, session, "meg", "*.ds"))])
            if len(runs) > 0:
                filenames = [os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_stimlock-meta.hdf') for run in runs]
                meta_data = pd.concat(pymeg.preprocessing.load_meta(filenames))
                meta_ses.append( meta_data )
        meta = pd.concat(meta_ses, axis=0)
        
        print( )
        print( )
        print(subj)
        meta_stim = (meta.query("session=='A'").stim_meg>0) # correct key press
        meta_resp = (meta.query("session=='A'").resp_meg>0) # subject key press
        print(np.mean(meta_stim==meta_resp))
        meta_stim = (meta.query("session=='B'").stim_meg<0) # correct key press
        meta_resp = (meta.query("session=='B'").resp_meg>0) # subject key press
        print(np.mean(meta_stim==meta_resp))
        
        min_rt.append(min(meta['rt'] - meta['rt_meg']))
        max_rt.append(max(meta['rt'] - meta['rt_meg']))
    
    print('min RT diff: {} ms'.format(round(min(np.array(min_rt))*1000,1)))
    print('max RT diff: {} ms'.format(round(max(np.array(max_rt))*1000,1)))
    
def select_sensors(subj):
    
    for session in ["A", "B"]:
        
        # load meta data:
        meta_filename_stim = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format('stimlock'))
        meta_filename_resp = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format('resplock'))
        meta_data_stim = pymeg.preprocessing.load_meta([meta_filename_stim])[0]
        meta_data_resp = pymeg.preprocessing.load_meta([meta_filename_resp])[0]
        
        # fix meta data:
        meta_data_stim["all"] = 1
        meta_data_stim["left"] = (meta_data_stim["resp_meg"] < 0).astype(int)
        meta_data_stim["right"] = (meta_data_stim["resp_meg"] > 0).astype(int)
        meta_data_resp["all"] = 1
        meta_data_resp["left"] = (meta_data_resp["resp_meg"] < 0).astype(int)
        meta_data_resp["right"] = (meta_data_resp["resp_meg"] > 0).astype(int)

        for sens in ['pos', 'left', 'right']:
            if sens == 'pos':
                nr_sensors = 25
                f = [70, 100]
                t = [0.3, 1]
            else:
                nr_sensors = 25
                f = [12, 36]
                t = [-0.5, 0]
    
            # pick sensors:
            sensors = pick_channels[sens](channels)
            print(len(sensors))
            
            # load TFR data:
            tfr_filename_stim = os.path.join(data_folder, "epochs", subj, session, "{}-tfr.hdf".format('stimlock'))
            tfr_filename_resp = os.path.join(data_folder, "epochs", subj, session, "{}-tfr.hdf".format('resplock'))
            tfr_data_stim = pymeg.tfr.get_tfrs([tfr_filename_stim], freq=(foi[0], foi[-1]), channel=sensors, tmin=-0.5, tmax=1.5, baseline=None)
            tfr_data_resp = pymeg.tfr.get_tfrs([tfr_filename_resp], freq=(foi[0], foi[-1]), channel=sensors, tmin=-0.9, tmax=0.4, baseline=None)
            
            if sens == 'pos':
                
                # get condition indices:
                trial_ind = meta_data_stim.loc[meta_data_stim["all"]==1, "hash"]
                
                # apply condition ind, and collapse across trials:
                tfr_data_condition = tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind, level='trial'),:].groupby(['freq', 'channel']).mean()
                
                # get baseline:
                baseline = baseline_per_sensor_get(tfr_data_condition, baseline=(-0.25, -0.15))
                
                # apply baseline:
                tfr_data_condition = baseline_per_sensor_apply(tfr_data_condition, baseline=baseline)
                
            else:
                
                # get condition indices:
                trial_ind_a = meta_data_stim.loc[meta_data_stim["right"]==1, "hash"]
                trial_ind_b = meta_data_stim.loc[meta_data_stim["left"]==1, "hash"]
                
                # apply condition ind, and collapse across trials:
                tfr_data_condition_a = tfr_data_resp.loc[tfr_data_resp.index.isin(trial_ind_a, level='trial'),:].groupby(['freq', 'channel']).mean()
                tfr_data_condition_b = tfr_data_resp.loc[tfr_data_resp.index.isin(trial_ind_b, level='trial'),:].groupby(['freq', 'channel']).mean()
                
                # get baseline:
                baseline_a = baseline_per_sensor_get(tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind_a, level='trial'),:].groupby(['freq', 'channel']).mean(), baseline=(-0.25, -0.15))
                baseline_b = baseline_per_sensor_get(tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind_b, level='trial'),:].groupby(['freq', 'channel']).mean(), baseline=(-0.25, -0.15))
                
                # apply baseline and make contast:
                tfr_data_condition = baseline_per_sensor_apply(tfr_data_condition_a, baseline=baseline_a) - \
                                     baseline_per_sensor_apply(tfr_data_condition_b, baseline=baseline_b)
                
            # collapse across freq:
            tfr_data_condition = tfr_data_condition.loc[(tfr_data_condition.index.get_level_values('freq')>=f[0]) & \
                                                            (tfr_data_condition.index.get_level_values('freq')<=f[1]),:].groupby(['channel']).mean()
            
            # collapse across time:
            tfr_data_condition = tfr_data_condition.loc[:,(tfr_data_condition.columns>=t[0]) & (tfr_data_condition.columns<=t[1])].mean(axis=1)
            
            # channels:
            if sens == 'left':
                best_sensors = tfr_data_condition.index[np.argsort(tfr_data_condition)][:nr_sensors]
            else:
                best_sensors = tfr_data_condition.index[np.argsort(tfr_data_condition)][-nr_sensors:]
            
            # save:
            np.save(os.path.join(data_folder, 'channels', '{}_{}_{}.npy'.format(sens, subj, session)), best_sensors)

            # delete:
            del tfr_data_stim
            del tfr_data_resp
                            
def make_tfr_contrasts(subj):
    
    for sens in ['pos', 'left', 'right']:
    # for sens in ['pos',]:
    # for sens in ['left', 'right']:
        
        # pick sensors:
        sensors = pick_channels[sens](channels)
        
        for session in ["A", "B"]:
            
            # filenames:
            meta_filename_stim = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format('stimlock'))
            meta_filename_resp = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format('resplock'))
            tfr_filename_stim = os.path.join(data_folder, "epochs", subj, session, "{}-tfr.hdf".format('stimlock'))
            tfr_filename_resp = os.path.join(data_folder, "epochs", subj, session, "{}-tfr.hdf".format('resplock'))
            
            if os.path.isfile(tfr_filename_stim):
                
                # load meta data:
                meta_data_stim = pymeg.preprocessing.load_meta([meta_filename_stim])[0]
                meta_data_resp = pymeg.preprocessing.load_meta([meta_filename_resp])[0]
                
                # fix meta data:
                meta_data_stim["all"] = 1
                meta_data_stim["left"] = (meta_data_stim["resp_meg"] < 0).astype(int)
                meta_data_stim["right"] = (meta_data_stim["resp_meg"] > 0).astype(int)
                meta_data_stim["hit"] = ((meta_data_stim["stimulus"] == 1) & (meta_data_stim["choice_a"] == 1)).astype(int)
                meta_data_stim["fa"] = ((meta_data_stim["stimulus"] == 0) & (meta_data_stim["choice_a"] == 1)).astype(int)
                meta_data_stim["miss"] = ((meta_data_stim["stimulus"] == 1) & (meta_data_stim["choice_a"] == 0)).astype(int)
                meta_data_stim["cr"] = ((meta_data_stim["stimulus"] == 0) & (meta_data_stim["choice_a"] == 0)).astype(int)
                meta_data_stim["left"] = (meta_data_stim["resp_meg"] < 0).astype(int)
                meta_data_stim["right"] = (meta_data_stim["resp_meg"] > 0).astype(int)
                meta_data_stim["pupil_h"] = (meta_data_stim["pupil_lp_d"] >= np.percentile(meta_data_stim["pupil_lp_d"], 60)).astype(int)
                meta_data_stim["pupil_l"] = (meta_data_stim["pupil_lp_d"] <= np.percentile(meta_data_stim["pupil_lp_d"], 40)).astype(int)
                
                meta_data_resp["all"] = 1
                meta_data_resp["left"] = (meta_data_resp["resp_meg"] < 0).astype(int)
                meta_data_resp["right"] = (meta_data_resp["resp_meg"] > 0).astype(int)
                meta_data_resp["hit"] = ((meta_data_resp["stimulus"] == 1) & (meta_data_resp["choice_a"] == 1)).astype(int)
                meta_data_resp["fa"] = ((meta_data_resp["stimulus"] == 0) & (meta_data_resp["choice_a"] == 1)).astype(int)
                meta_data_resp["miss"] = ((meta_data_resp["stimulus"] == 1) & (meta_data_resp["choice_a"] == 0)).astype(int)
                meta_data_resp["cr"] = ((meta_data_resp["stimulus"] == 0) & (meta_data_resp["choice_a"] == 0)).astype(int)
                meta_data_resp["left"] = (meta_data_resp["resp_meg"] < 0).astype(int)
                meta_data_resp["right"] = (meta_data_resp["resp_meg"] > 0).astype(int)
                meta_data_resp["pupil_h"] = (meta_data_resp["pupil_lp_d"] >= np.percentile(meta_data_resp["pupil_lp_d"], 60)).astype(int)
                meta_data_resp["pupil_l"] = (meta_data_resp["pupil_lp_d"] <= np.percentile(meta_data_resp["pupil_lp_d"], 40)).astype(int)
                
                # load TFR data:
                tfr_data_stim = pymeg.tfr.get_tfrs([tfr_filename_stim], freq=(foi[0], foi[-1]), channel=sensors, tmin=-0.5, tmax=1.5, baseline=None)
                tfr_data_resp = pymeg.tfr.get_tfrs([tfr_filename_resp], freq=(foi[0], foi[-1]), channel=sensors, tmin=-0.9, tmax=0.4, baseline=None)
                
                # collapse data across trial types:
                for n in [0,1,4]:
                # for n in [0]:
                    for c in ['all', 'hit', 'fa', 'miss', 'cr', 'left', 'right', 'pupil_h', 'pupil_l']:
                        
                        # get condition indices:
                        if n == 0:
                            
                            trial_ind_stim = meta_data_stim.loc[meta_data_stim[c]==1, "hash"]
                            trial_ind_resp = meta_data_resp.loc[meta_data_stim[c]==1, "hash"]
                            # trial_ind_stim = meta_data_stim.query('({}==1)'.format(c)).index.get_level_values('hash')
                            # trial_ind_resp = meta_data_resp.query('({}==1)'.format(c)).index.get_level_values('hash')
                        else:
                            trial_ind_stim = meta_data_stim.loc[(meta_data_stim[c]==1) & (meta_data_stim["noise"]==n), "hash"]
                            trial_ind_resp = meta_data_resp.loc[(meta_data_stim[c]==1) & (meta_data_stim["noise"]==n), "hash"]
                            # trial_ind_stim = meta_data_stim.query('(noise=={}) & ({}==1)'.format(n, c)).index.get_level_values('hash')
                            # trial_ind_resp = meta_data_resp.query('(noise=={}) & ({}==1)'.format(n, c)).index.get_level_values('hash')
                        if len(trial_ind_stim) > 0:
                            
                            # apply condition ind, and collapse across trials:
                            tfr_data_stim_condition = tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind_stim, level='trial'),:].groupby(['freq', 'channel']).mean()
                            tfr_data_resp_condition = tfr_data_resp.loc[tfr_data_resp.index.isin(trial_ind_resp, level='trial'),:].groupby(['freq', 'channel']).mean()
                            
                            # subselect sensors:
                            if session == 'A':
                                best_sensors = np.load(os.path.join(data_folder, 'channels', '{}_{}_{}.npy'.format(sens, subj, 'B')),)
                            if session == 'B':
                                best_sensors = np.load(os.path.join(data_folder, 'channels', '{}_{}_{}.npy'.format(sens, subj, 'A')),)
                            tfr_data_stim_condition = tfr_data_stim_condition.loc[tfr_data_stim_condition.index.isin(best_sensors, level='channel'),:]
                            tfr_data_resp_condition = tfr_data_resp_condition.loc[tfr_data_resp_condition.index.isin(best_sensors, level='channel'),:]
                            
                            # get baseline:
                            baseline = baseline_per_sensor_get(tfr_data_stim_condition, baseline=(-0.25, -0.15))
                            
                            # apply baseline, and collapse across sensors:
                            tfr_data_stim_condition = baseline_per_sensor_apply(tfr_data_stim_condition, baseline=baseline).groupby(['freq']).mean()
                            tfr_data_resp_condition = baseline_per_sensor_apply(tfr_data_resp_condition, baseline=baseline).groupby(['freq']).mean()
                            
                            # save:
                            tfr_data_stim_condition.to_hdf(os.path.join(data_folder, 'tfr', '{}_{}_{}_{}_{}_{}.hdf'.format(sens, 'stim', subj, n, c, session)), 'tfr')
                            tfr_data_resp_condition.to_hdf(os.path.join(data_folder, 'tfr', '{}_{}_{}_{}_{}_{}.hdf'.format(sens, 'resp', subj, n, c, session)), 'tfr')
                            
                            # # plot:
                            # times = np.array(tfr_data_stim_condition.columns, dtype=float)
                            # freqs = np.array(np.unique(tfr_data_stim_condition.index.get_level_values('freq')), dtype=float)
                            # X = np.array(tfr_data_stim_condition)
                            # fig = plt.figure()
                            # ax = fig.add_subplot(111)
                            # cax = ax.pcolormesh(times, freqs, X, vmin=-15, vmax=15, cmap='jet')
                            # fig.savefig('test.pdf')

if __name__ == '__main__':

    pick_channels = dict(
                    ant=lambda x: [ch for ch in x if ch.startswith('MLC') or ch.startswith('MRC') or ch.startswith('MLF') or ch.startswith('MRF') or ch.startswith('MZC') or ch.startswith('MZF') \
                                                  or (ch in ['MLT{}-3705'.format(c) for c in [67, 13, 23, 33, 12, 22, 42, 51, 11, 32, 41, 21, 31]]) \
                                                  or (ch in ['MRT{}-3705'.format(c) for c in [67, 13, 23, 33, 12, 22, 42, 51, 11, 32, 41, 21, 31]])],
                    pos=lambda x: [ch for ch in x if ch.startswith('MLO') or ch.startswith('MRO') or ch.startswith('MLP') or ch.startswith('MRP') or ch.startswith('MZO') or ch.startswith('MZP') \
                                                  or (ch in ['MLT{}-3705'.format(c) for c in [14, 24, 34, 43, 52, 15, 25, 35, 44, 53, 16, 26, 36, 45, 54, 27, 37, 46, 55, 47, 56, 57]]) \
                                                  or (ch in ['MRT{}-3705'.format(c) for c in [14, 24, 34, 43, 52, 15, 25, 35, 44, 53, 16, 26, 36, 45, 54, 27, 37, 46, 55, 47, 56, 57]])],
                    left=lambda x: [ch for ch in x if ch.startswith('MLO') or ch.startswith('MLP') or ch.startswith('MLT') or ch.startswith('MLC') or ch.startswith('MLF')],
                    right=lambda x: [ch for ch in x if ch.startswith('MRO') or ch.startswith('MRP') or ch.startswith('MRT') or ch.startswith('MRC') or ch.startswith('MRF')],
                    all=lambda x: [ch for ch in x if ch.startswith('M')],
                    occ=lambda x: [ch for ch in x if ch.startswith('MLO') or ch.startswith('MRO')],
                    par=lambda x: [ch for ch in x if ch.startswith('MLP') or ch.startswith('MRP')],
                    cent=lambda x: [ch for ch in x if ch.startswith('MLC') or ch.startswith('MRC') or ch.startswith('MZC')],
                    centr=lambda x: [ch for ch in x if ch.startswith('MRC') or ch.startswith('MRP')],
                    centl=lambda x: [ch for ch in x if ch.startswith('MLC') or ch.startswith('MLP')],
                    frontal=lambda x: [ch for ch in x if ch.startswith('MLF') or ch.startswith('MRF') or ch.startswith('MZF')],
                    temp=lambda x: [ch for ch in x if ch.startswith('MLT') or ch.startswith('MRT')])

    # subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15',         'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23',]

    make_contrasts = False
    do_plots = True

    # rt_diffs(subjects)
    if make_contrasts:
        # n_jobs = 2
        # _ = Parallel(n_jobs=n_jobs)(delayed(make_tfr_contrasts)(subj) for subj in subjects)
        
        # serial:
        for subj in subjects:
            print(subj)
            select_sensors(subj)
            make_tfr_contrasts(subj)
            
    if do_plots:
        for sens in ['pos', 'lat',]:
        # for sens in ['pos']:
        # for sens in []:
            for n in [0,1,4]:
            # for n in [0]:
            # for n in [0,]:
                
                if sens == 'pos':
                    contrasts = ['all', 'stimulus', 'choice_a', 'pupil', 'cor_choice_a_pupil',]
                    # contrasts = ['cor_choice_a_pupil',]
                else:
                    contrasts = ['hand',]
                
                for c in contrasts:
                # for c in ['all',]:
                    # for perm in [False, True]:
                    for perm in [True]:
                        
                        fig = plt.figure(figsize=(4,2))
                        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1.25, 1])
                        
                        fig_s = plt.figure(figsize=(16,np.ceil(len(subjects)/2.0)))
                        plot_nr_stim = 1
                        plot_nr_resp = 2
                        for a, tl in enumerate(['stim', 'resp',]):
                            
                            tfr_group_a = []
                            tfr_group_b = []
                            for i, subj in enumerate(subjects):
                                
                                trial_types = ['hit', 'fa', 'miss', 'cr', 'right', 'left', 'all', 'pupil_h', 'pupil_l']
                                if sens == 'lat':
                                    tfrs = []
                                    for tt in trial_types:
                                        tfr_session = []
                                        for session in ['A', 'B']:
                                            try:
                                                tfr_session.append( pd.read_hdf(os.path.join(data_folder, 'tfr', '{}_{}_{}_{}_{}_{}.hdf'.format('left', tl, subj, n, tt, session)), 'tfr') - \
                                                                    pd.read_hdf(os.path.join(data_folder, 'tfr', '{}_{}_{}_{}_{}_{}.hdf'.format('right', tl, subj, n, tt, session)), 'tfr') )
                                            except:
                                                pass
                                        tfrs.append( pd.concat(tfr_session).groupby('freq').mean() )
                                else:
                                    tfrs = []
                                    for tt in trial_types:
                                        tfr_session = []
                                        for session in ['A', 'B']:
                                            try:
                                                tfr_session.append( pd.read_hdf(os.path.join(data_folder, 'tfr', '{}_{}_{}_{}_{}_{}.hdf'.format(sens, tl, subj, n, tt, session)), 'tfr') )
                                            except:
                                                pass
                                        tfrs.append( pd.concat(tfr_session).groupby('freq').mean() )
                                       
                                if c == 'all':
                                    # tfr = (tfrs[0]+tfrs[1]+tfrs[2]+tfrs[3]) / 4.0
                                    tfr = tfrs[6]
                                if c == 'stimulus':
                                    tfr = (tfrs[0]+tfrs[2]) - (tfrs[1]+tfrs[3])   
                                if c == 'choice_a':
                                    tfr = (tfrs[0]+tfrs[1]) - (tfrs[2]+tfrs[3])    
                                if c == 'hand':
                                    tfr = tfrs[4]-tfrs[5] 
                                if c == 'pupil':
                                    tfr = tfrs[6]-tfrs[7]
                                if c == 'cor_choice_a_pupil':
                                    tfr = (tfrs[0]+tfrs[1]) - (tfrs[2]+tfrs[3])
                                    tfr_b = tfrs[6]-tfrs[7]
                                    tfr_b["subj_idx"] = i
                                    tfr_group_b.append( tfr_b )
                                    
                                tfr["subj_idx"] = i
                                tfr_group_a.append( tfr )
                            tfr = pd.concat(tfr_group_a, axis=0)
                            tfr = tfr.set_index(['subj_idx'], append=True)
                            
                            if 'cor' in c:
                                tfr_b = pd.concat(tfr_group_b, axis=0)
                                tfr_b = tfr_b.set_index(['subj_idx'], append=True)
                                
                            # cmap:
                            # cmap = 'RdYlBu_r'
                            
                            from matplotlib.colors import LinearSegmentedColormap
                            cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)
                            
                            # time:
                            if tl  == 'stim':
                                time_cutoff = (-0.1, 1.3)
                                xlabel = 'Time from stimulus (s)'
                            if tl  == 'resp':
                                time_cutoff = (-0.7, 0.2)
                                xlabel = 'Time from response (s)'
                    
                            # vmin vmax:
                            if c == 'all':
                                vmin, vmax = (-12.5, 12.5)
                            elif 'cor' in c:
                                vmin, vmax = (-0.75, 0.75)
                            else:
                                vmin, vmax = (-5, 5)
                    
                            # threshold:
                            threshold = 0.05
                            
                            # plot:
                            times = np.array(tfr.columns, dtype=float)
                            freqs = np.array(np.unique(tfr.index.get_level_values('freq')), dtype=float)
                            time_ind = (times>time_cutoff[0]) & (times<time_cutoff[1])
                            time_ind = (times>time_cutoff[0]) & (times<time_cutoff[1])
                            if 'cor' in c:
                                X_a = np.stack([tfr.loc[tfr.index.isin(i, level='subj_idx'), time_ind].values for i in enumerate(subjects)])
                                X_b = np.stack([tfr_b.loc[tfr_b.index.isin(i, level='subj_idx'), time_ind].values for i in enumerate(subjects)])
                                
                                X = np.zeros((X_a.shape[1], X_a.shape[2]))
                                for j in range(X_a.shape[1]):
                                    for k in range(X_a.shape[2]):
                                        X[j,k] = sp.stats.pearsonr(X_a[:,j,k], X_b[:,j,k])[0]
                            else:
                                X = np.stack([tfr.loc[tfr.index.isin(i, level='subj_idx'), time_ind].values for i in enumerate(subjects)])
                            
                            # shell()
                            
                            # grand average plot:
                            ax = fig.add_subplot(gs[a]) 
                            if 'cor' in c:
                                cax = ax.pcolormesh(times[time_ind], freqs, X, vmin=vmin, vmax=vmax, cmap=cmap)
                            else:
                                cax = ax.pcolormesh(times[time_ind], freqs, X.mean(axis=0), vmin=vmin, vmax=vmax, cmap=cmap)
                                
                                if tl  == 'stim':
                                    test_data = X[:,:,times[time_ind]>0]
                                    times_test_data = times[time_ind][times[time_ind]>0]
                                else:
                                    test_data = X.copy()
                                    times_test_data = times[time_ind]
                                
                                T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(test_data, threshold={'start':0, 'step':0.2}, 
                                                                                                                connectivity=None, tail=0, n_permutations=1000, n_jobs=6)
                                sig = cluster_p_values.reshape((test_data.shape[1], test_data.shape[2]))
                                
                                # T_obs2 = sp.stats.ttest_1samp(X, 0)[0]
                                # fig = plt.figure()
                                # # plt.pcolormesh(times[time_ind], freqs, sig, vmin=0, vmax=1, cmap=cmap)
                                # plt.pcolormesh(times[time_ind], freqs, T_obs2, vmin=-10, vmax=10, cmap=cmap)
                                # fig.savefig('test.pdf')
                                
                                ax.contour(times_test_data, freqs, sig, (threshold,), linewidths=0.5, colors=('black'))
                            ax.axvline(0, ls='--', lw=0.75, color='black',)
                            if tl  == 'stim':
                                ax.axvline(1, ls='--', lw=0.75, color='black',)
                                if (sens == 'pos') & (c == 'all'):
                                    ax.add_patch(matplotlib.patches.Rectangle(xy=(0.3,70), width=0.7, height=30, lw=0.5, ls='--', color='w', fill=False))
                            if tl  == 'resp':
                                if (sens == 'lat') & (c == 'hand'):
                                    ax.add_patch(matplotlib.patches.Rectangle(xy=(-0.5,12), width=0.5, height=24, lw=0.5, ls='--', color='w', fill=False))
                            
                            ax.set_xlabel(xlabel)
                            if a == 0:
                                ax.set_ylabel('Frequency (Hz)')
                                ax.set_title('{} contrast'.format(c))
                            elif a == 1:
                                ax.set_title('N = {}'.format(len(subjects)))
                                fig.colorbar(cax, ticks=[vmin, 0, vmax])
                                ax.tick_params(labelleft='off') 
                            
                            if not 'cor' in c:
                            
                                # separately per subject:
                                for i in range(len(subjects)):
                                    if tl  == 'stim':
                                        ax_s = fig_s.add_subplot(np.ceil(len(subjects)/4.0),8,plot_nr_stim)
                                    elif tl  == 'resp':
                                        ax_s = fig_s.add_subplot(np.ceil(len(subjects)/4.0),8,plot_nr_resp)
                                    cax_s = ax_s.pcolormesh(times[time_ind], freqs, X[i,:,:], vmin=-15, vmax=15, cmap=cmap)
                                    ax_s.axvline(0, ls='--', lw=0.75, color='black',)
                                    if tl  == 'stim':
                                        ax_s.axvline(1, ls='--', lw=0.75, color='black',)
                                    ax_s.set_title('S{}'.format(subjects[i][-2:]))
                                    ax_s.set_xlabel(xlabel)
                                    ax_s.set_ylabel('Frequency (Hz)')
                                    if tl  == 'stim':
                                        plot_nr_stim += 2
                                    elif tl  == 'resp':
                                        plot_nr_resp += 2
                        
                        fig.tight_layout()
                        fig.savefig(os.path.join(fig_folder, 'sensor_level', '{}_{}_{}_{}.pdf'.format(sens, n, c, 0)))
                        
                        fig_s.tight_layout()
                        fig_s.savefig(os.path.join(fig_folder, 'sensor_level', 'subjects', '{}_{}_{}.png'.format(sens, n, c)))