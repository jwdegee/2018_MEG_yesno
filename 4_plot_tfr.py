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
# data_folder = '\\\\discovery1\\bcm-neuro-mcginley\\JW\\data'
# fig_folder = '\\\\discovery1\\bcm-neuro-mcginley\\JW\\figures'
data_folder = '/media/external3/JW/data/'
fig_folder = '/media/external3/JW/figures/'

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

def load_meta_data(subj, session, timelock, data_folder):

    # load:
    meta_data_filename = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format(timelock))
    meta_data = pymeg.preprocessing.load_meta([meta_data_filename])[0]

    # add columns:
    meta_data["all"] = 1
    meta_data["left"] = (meta_data["resp_meg"] < 0).astype(int)
    meta_data["right"] = (meta_data["resp_meg"] > 0).astype(int)
    meta_data["hit"] = ((meta_data["stimulus"] == 1) & (meta_data["choice_a"] == 1)).astype(int)
    meta_data["fa"] = ((meta_data["stimulus"] == 0) & (meta_data["choice_a"] == 1)).astype(int)
    meta_data["miss"] = ((meta_data["stimulus"] == 1) & (meta_data["choice_a"] == 0)).astype(int)
    meta_data["cr"] = ((meta_data["stimulus"] == 0) & (meta_data["choice_a"] == 0)).astype(int)
    meta_data["pupil_h"] = (meta_data["pupil_lp_d"] >= np.percentile(meta_data["pupil_lp_d"], 60)).astype(int)
    meta_data["pupil_l"] = (meta_data["pupil_lp_d"] <= np.percentile(meta_data["pupil_lp_d"], 40)).astype(int)
    meta_data["conf_h"] = (meta_data.confidence > 1).astype(int)
    meta_data["conf_l"] = (meta_data.confidence <= 1).astype(int)
    meta_data["noise_fast"] = (meta_data.noise == 1).astype(int)
    meta_data["noise_slow"] = (meta_data.noise == 4).astype(int)

    meta_data["noise_fast_left"] = ((meta_data["left"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_right"] = ((meta_data["right"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_hit"] = ((meta_data["hit"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_fa"] = ((meta_data["fa"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_miss"] = ((meta_data["miss"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_cr"] = ((meta_data["cr"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_pupil_h"] = ((meta_data["pupil_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_pupil_l"] = ((meta_data["pupil_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_conf_h"] = ((meta_data["conf_h"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)
    meta_data["noise_fast_conf_l"] = ((meta_data["conf_l"] == 1) & (meta_data["noise_fast"] == 1)).astype(int)

    meta_data["noise_slow_left"] = ((meta_data["left"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_right"] = ((meta_data["right"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_hit"] = ((meta_data["hit"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_fa"] = ((meta_data["fa"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_miss"] = ((meta_data["miss"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_cr"] = ((meta_data["cr"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_pupil_h"] = ((meta_data["pupil_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_pupil_l"] = ((meta_data["pupil_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_conf_h"] = ((meta_data["conf_h"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)
    meta_data["noise_slow_conf_l"] = ((meta_data["conf_l"] == 1) & (meta_data["noise_slow"] == 1)).astype(int)

    return meta_data

def select_sensors(subj, baseline_time=(-0.25, -0.15)):
    
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
                baseline_a = baseline_per_sensor_get(tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind_a, level='trial'),:].groupby(['freq', 'channel']).mean(), baseline=baseline_time)
                baseline_b = baseline_per_sensor_get(tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind_b, level='trial'),:].groupby(['freq', 'channel']).mean(), baseline=baseline_time)
                
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
            np.save(os.path.join(data_folder, 'sensor_level', 'channels', '{}_{}_{}.npy'.format(sens, subj, session)), best_sensors)

            # delete:
            del tfr_data_stim
            del tfr_data_resp
                            
def make_tfr_contrasts(subj, baseline_time=(-0.25, -0.15)):
    
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
                meta_data = load_meta_data(subj, session, 'stimlock', data_folder)
                               
                # load TFR data:
                tfr_data_stim = pymeg.tfr.get_tfrs([tfr_filename_stim], freq=(foi[0], foi[-1]), channel=sensors, tmin=-0.5, tmax=1.5, baseline=None)
                tfr_data_resp = pymeg.tfr.get_tfrs([tfr_filename_resp], freq=(foi[0], foi[-1]), channel=sensors, tmin=-0.9, tmax=0.4, baseline=None)
                
                # collapse data across trial types:
                for n in [0,1,4]:
                # for n in [0]:
                    for c in ['all', 'hit', 'fa', 'miss', 'cr', 'left', 'right', 'pupil_h', 'pupil_l', 'conf_h', 'conf_l']:
                        
                        # get condition indices:
                        if n == 0:
                            trial_ind = meta_data.loc[meta_data[c]==1, "hash"]
                        else:
                            trial_ind = meta_data.loc[(meta_data[c]==1) & (meta_data["noise"]==n), "hash"]
                        if len(trial_ind) > 0:
                            
                            # apply condition ind, and collapse across trials:
                            tfr_data_stim_condition = tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind, level='trial'),:].groupby(['freq', 'channel']).mean()
                            tfr_data_resp_condition = tfr_data_resp.loc[tfr_data_resp.index.isin(trial_ind, level='trial'),:].groupby(['freq', 'channel']).mean()
                            
                            # subselect sensors:
                            if session == 'A':
                                best_sensors = np.load(os.path.join(data_folder, 'sensor_level', 'channels', '{}_{}_{}.npy'.format(sens, subj, 'B')),)
                            if session == 'B':
                                best_sensors = np.load(os.path.join(data_folder, 'sensor_level', 'channels', '{}_{}_{}.npy'.format(sens, subj, 'A')),)
                            tfr_data_stim_condition = tfr_data_stim_condition.loc[tfr_data_stim_condition.index.isin(best_sensors, level='channel'),:]
                            tfr_data_resp_condition = tfr_data_resp_condition.loc[tfr_data_resp_condition.index.isin(best_sensors, level='channel'),:]
                            
                            # get baseline:
                            baseline = baseline_per_sensor_get(tfr_data_stim_condition, baseline=baseline_time)
                            
                            # apply baseline, and collapse across sensors:
                            tfr_data_stim_condition = baseline_per_sensor_apply(tfr_data_stim_condition, baseline=baseline).groupby(['freq']).mean()
                            tfr_data_resp_condition = baseline_per_sensor_apply(tfr_data_resp_condition, baseline=baseline).groupby(['freq']).mean()
                            
                            # save:
                            tfr_data_stim_condition.to_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format(sens, 'stim', subj, n, c, session)), 'tfr')
                            tfr_data_resp_condition.to_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format(sens, 'resp', subj, n, c, session)), 'tfr')
                            
                            # # plot:
                            # times = np.array(tfr_data_stim_condition.columns, dtype=float)
                            # freqs = np.array(np.unique(tfr_data_stim_condition.index.get_level_values('freq')), dtype=float)
                            # X = np.array(tfr_data_stim_condition)
                            # fig = plt.figure()
                            # ax = fig.add_subplot(111)
                            # cax = ax.pcolormesh(times, freqs, X, vmin=-15, vmax=15, cmap='jet')
                            # fig.savefig('test.pdf')

def make_single_trial_slopes(subj, baseline_time=(-0.25, -0.15)):

    f = [12, 36]
    t = [-0.5, 0]

    for sens in ['left', 'right']:
                
        # pick sensors:
        sensors = pick_channels[sens](channels)
        
        for session in ["A", "B"]:
            
            # filenames:
            tfr_filename_stim = os.path.join(data_folder, "epochs", subj, session, "{}-tfr.hdf".format('stimlock'))
            tfr_filename_resp = os.path.join(data_folder, "epochs", subj, session, "{}-tfr.hdf".format('resplock'))
            
            if os.path.isfile(tfr_filename_stim):
                
                # load meta data:
                meta_data = load_meta_data(subj, session, 'stimlock', data_folder)
                               
                # load TFR data:
                tfr_data_stim = pymeg.tfr.get_tfrs([tfr_filename_stim], freq=(foi[0], foi[-1]), channel=sensors, tmin=-0.5, tmax=1.5, baseline=None)
                tfr_data_resp = pymeg.tfr.get_tfrs([tfr_filename_resp], freq=(foi[0], foi[-1]), channel=sensors, tmin=-0.9, tmax=0.4, baseline=None)
                
                # # same trials:
                # trial_ind = np.intersect1d(tfr_data_stim.index.get_level_values('trial'), tfr_data_resp.index.get_level_values('trial'))
                # tfr_data_stim_condition = tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind, level='trial'),:].groupby(['freq', 'channel']).mean()
                # tfr_data_resp_condition = tfr_data_resp.loc[tfr_data_resp.index.isin(trial_ind, level='trial'),:].groupby(['freq', 'channel']).mean()

                # # subselect sensors:
                # if session == 'A':
                #     best_sensors = np.load(os.path.join(data_folder, 'sensor_level', 'channels', '{}_{}_{}.npy'.format(sens, subj, 'B')),)
                # if session == 'B':
                #     best_sensors = np.load(os.path.join(data_folder, 'sensor_level', 'channels', '{}_{}_{}.npy'.format(sens, subj, 'A')),)
                # tfr_data_stim_condition = tfr_data_stim_condition.loc[tfr_data_stim_condition.index.isin(best_sensors, level='channel'),:]
                # tfr_data_resp_condition = tfr_data_resp_condition.loc[tfr_data_resp_condition.index.isin(best_sensors, level='channel'),:]
                
                # # get baseline:
                # baseline = baseline_per_sensor_get(tfr_data_stim_condition, baseline=baseline_time)
                
                # # apply baseline, and collapse across sensors:
                # tfr_data_stim_condition = baseline_per_sensor_apply(tfr_data_stim_condition, baseline=baseline).groupby(['freq']).mean()
                # tfr_data_resp_condition = baseline_per_sensor_apply(tfr_data_resp_condition, baseline=baseline).groupby(['freq']).mean()


                # from matplotlib.colors import LinearSegmentedColormap
                # cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)

                # tfr = tfr_data_resp_condition.copy()
                # times = np.array(tfr.columns, dtype=float)
                # freqs = np.array(np.unique(tfr.index.get_level_values('freq')), dtype=float)
                # time_ind = (times>-0.5) & (times<0.3)
                # vmin = -5
                # vmax = 5
                # plt.pcolormesh(times[time_ind], freqs, tfr.values[:,time_ind], vmin=vmin, vmax=vmax, cmap=cmap)

                # same trials:
                trial_ind = np.intersect1d(tfr_data_stim.index.get_level_values('trial'), tfr_data_resp.index.get_level_values('trial'))
                tfr_data_stim_condition = tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind, level='trial'),:]
                tfr_data_resp_condition = tfr_data_resp.loc[tfr_data_resp.index.isin(trial_ind, level='trial'),:]

                # subselect sensors:
                if session == 'A':
                    best_sensors = np.load(os.path.join(data_folder, 'sensor_level', 'channels', '{}_{}_{}.npy'.format(sens, subj, 'B')),)
                if session == 'B':
                    best_sensors = np.load(os.path.join(data_folder, 'sensor_level', 'channels', '{}_{}_{}.npy'.format(sens, subj, 'A')),)
                tfr_data_stim_condition = tfr_data_stim_condition.loc[tfr_data_stim_condition.index.isin(best_sensors, level='channel'),:]
                tfr_data_resp_condition = tfr_data_resp_condition.loc[tfr_data_resp_condition.index.isin(best_sensors, level='channel'),:]
                
                # baseline:
                time = tfr_data_stim_condition.columns.get_level_values('time').values.astype(float)
                id_base =  (time > baseline_time[0]) & (time < baseline_time[1])
                baseline = np.atleast_2d(tfr_data_stim_condition.loc[:, id_base].mean(axis=1)).T
                a = tfr_data_resp_condition.copy()
                a.loc[:,:] = (a.values - baseline) #/ baseline  * 100

                # collapse across sensors:
                a = a.groupby(['trial', 'freq']).mean()

                # collapse across freq:
                a = a.loc[(a.index.get_level_values('freq')>=f[0]) & (a.index.get_level_values('freq')<=f[1]),:].groupby(['trial']).mean()



                trial_ind = np.unique(a.index.get_level_values('trial'))
                meta_data = meta_data.loc[meta_data['hash'].isin(trial_ind),:]
                # time_ind = (a.columns >= -0.5) & (a.columns <= 0.1)
                time_ind = (a.columns >= -0.1) & (a.columns <= 1.1)
                plt.plot(a.loc[np.array(meta_data.right==1),time_ind].mean(axis=0))
                plt.plot(a.loc[np.array(meta_data.left==1),time_ind].mean(axis=0))


                
                # save:
                tfr_data_resp_condition.to_hdf(os.path.join(data_folder, 'sensor_level', 'beta_slopes', '{}_{}_{}_{}.hdf'.format(sens, 'stim', subj, session)), 'tfr')
                
                # # plot:
                # times = np.array(tfr_data_stim_condition.columns, dtype=float)
                # freqs = np.array(np.unique(tfr_data_stim_condition.index.get_level_values('freq')), dtype=float)
                # X = np.array(tfr_data_stim_condition)
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # cax = ax.pcolormesh(times, freqs, X, vmin=-15, vmax=15, cmap='jet')
                # fig.savefig('test.pdf')



def plot_tfr(tfr, time_cutoff, vmin, vmax, tl, cluster_correct=False, threshold=0.05, plot_colorbar=False, weigthed_mean=False, weights=None, ax=None):

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
    if weigthed_mean:
        fas = [min(weights[s]) for s in weights.keys() if len(weights[s]) > 0]
        fas = fas / max(fas)
        for i, fa in enumerate(fas):
            X[i,:,:] = X[i,:,:] * fa
        mean_tfr = X.sum(axis=0) / sum(fas)
    else:
        mean_tfr = X.mean(axis=0)
    cax = ax.pcolormesh(times[time_ind], freqs, mean_tfr, vmin=vmin, vmax=vmax, cmap=cmap)
        
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
                                                                                        connectivity=None, tail=0, n_permutations=1000, n_jobs=10)
            sig = cluster_p_values.reshape((test_data.shape[1], test_data.shape[2]))
            ax.contour(times_test_data, freqs, sig, (threshold,), linewidths=0.5, colors=('black'))
        except:
            pass
    ax.axvline(0, ls='--', lw=0.75, color='black',)
    if plot_colorbar:
        plt.colorbar(cax, ticks=[vmin, 0, vmax])
    return ax

def select_sessions(subjects, cutoff=5):
    
    subjs = []
    sess = []
    fas = []
    for subj in subjects:
        print(subj)
        for session in ['A', 'B']:
            try:
                meta = load_meta_data(subj, session, 'stimlock', data_folder)
                fa = meta.loc[meta['noise']==4, 'fa'].sum()
                subjs.append(subj)
                sess.append(session)
                fas.append(fa)
            except:
                pass
    subjs = np.array(subjs)
    sess = np.array(sess)
    fas = np.array(fas)

    print('keep {} out of {} subjects'.format(len(np.unique(subjs[fas>=cutoff])), len(np.unique(subjs))))
    print('keep {} out of {} sessions'.format(sum(fas>=cutoff), len(sess)))

    fig = plt.figure(figsize=(12,2))
    ax = fig.add_subplot(121)
    ax.errorbar(np.arange(len(fas)), fas, fmt='-o')
    ax.set_xticks(np.arange(len(fas)))
    ax.set_xticklabels(labels=subjs, rotation=45)
    ax.axhline(cutoff, ls='--', color='r')
    plt.ylabel('# FAs in slow noise condition')
    ax = fig.add_subplot(122)
    ax.hist(fas, bins=50)
    ax.axvline(cutoff, ls='--', color='r')
    plt.xlabel('# FAs in slow noise condition')
    plt.ylabel('# sessions')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_folder, 'false_alarms.pdf'))

    which_sessions = {}
    false_alarms = {}
    for subj in np.unique(subjs):
        session_to_analyse = list(sess[(subjs==subj)&(fas>=cutoff)])
        nr_fas = list(fas[(subjs==subj)&(fas>=cutoff)])
        which_sessions.update({subj: session_to_analyse})
        false_alarms.update({subj: nr_fas})
    print(which_sessions)
    return which_sessions, false_alarms

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
    cluster_correct = False

    which_sessions, false_alarms = select_sessions(subjects, cutoff=5)

    # rt_diffs(subjects)
    if make_contrasts:
        # n_jobs = 2
        # _ = Parallel(n_jobs=n_jobs)(delayed(make_tfr_contrasts)(subj) for subj in subjects)
        
        # serial:
        for subj in subjects:
            print(subj)
            # select_sensors(subj)
            # make_tfr_contrasts(subj)
            make_single_trial_slopes(subj)
            
    if do_plots:
        
        # subj = 'jw02'
        # session = 'A'

        # meta_data = load_meta_data(subj, session, 'stimlock', data_folder)
        # left = pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'beta_slopes', '{}_{}_{}_{}.hdf'.format('left', 'stim', subj, session)), 'tfr')
        # right = pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'beta_slopes', '{}_{}_{}_{}.hdf'.format('right', 'stim', subj, session)), 'tfr')
        
        # trial_ind = np.unique(np.concatenate((left.index.get_level_values('trial'), right.index.get_level_values('trial'))))
        # meta_data = meta_data.loc[meta_data['hash'].isin(trial_ind),:]
        # left = left.loc[left.index.isin(trial_ind, level='trial'),:]
        # right = right.loc[right.index.isin(trial_ind, level='trial'),:]

        # t = (-0.6, 0.1)
        # left = left.loc[:,(left.columns>=t[0]) & (left.columns<=t[1])]
        # right = right.loc[:,(right.columns>=t[0]) & (right.columns<=t[1])]


        # if meta_data.shape[0] != left.shape[0]:
        #     raise 
        #        
        # lat = left - right
        # plt.plot(left.loc[np.array(meta_data.right==1),:].mean(axis=0))
        # plt.plot(left.loc[np.array(meta_data.left==1),:].mean(axis=0))
        # 
        # plt.plot(right.loc[np.array(meta_data.right==1),:].mean(axis=0))
        # plt.plot(right.loc[np.array(meta_data.left==1),:].mean(axis=0))
        # 
        # plt.plot(left.mean(axis=0))
        # plt.plot(left.loc[np.array(meta_data.left==1),:].mean(axis=0))

        fig = plt.figure(figsize=(8,8))
        plot_nr = 1
        for n in [1,4]:
            for tl in ['stim', 'resp']:
                tfrs_hit = []
                tfrs_fa = []
                tfrs_miss = []
                tfrs_cr = []
                tfr_group_b = []
                for i, subj in enumerate(subjects):
                    print(subj)
                    trial_types = ['hit', 'fa', 'miss', 'cr', 'right', 'left', 'all', 'pupil_h', 'pupil_l', 'conf_h', 'conf_l']
                    # trial_types = ['hit', 'fa', 'miss', 'cr', 'right', 'left', 'all', 'pupil_h', 'pupil_l',]
                    if len(which_sessions[subj]) > 0:
                        tfrs = []
                        for tt in trial_types:
                            tfr_session = []
                            for session in which_sessions[subj]:
                                # try:
                                if session == 'A':
                                    tfr_session.append( pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format('right', tl, subj, n, tt, session)), 'tfr') - \
                                                        pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format('left', tl, subj, n, tt, session)), 'tfr') )
                                elif session == 'B':
                                    tfr_session.append( pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format('left', tl, subj, n, tt, session)), 'tfr') - \
                                                        pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format('right', tl, subj, n, tt, session)), 'tfr') )
                            tfrs.append( pd.concat(tfr_session).groupby('freq').mean() )
                        tfr_hit = tfrs[0]
                        tfr_hit["subj"] = i
                        tfr_fa = tfrs[1]
                        tfr_fa["subj"] = i
                        tfr_miss = tfrs[2]
                        tfr_miss["subj"] = i
                        tfr_cr = tfrs[3]
                        tfr_cr["subj"] = i
                        tfrs_hit.append( tfr_hit )
                        tfrs_fa.append( tfr_fa )
                        tfrs_miss.append( tfr_miss )
                        tfrs_cr.append( tfr_cr )
                tfr_hit = pd.concat(tfrs_hit, axis=0)
                tfr_hit = tfr_hit.set_index(['subj'], append=True)
                tfr_fa = pd.concat(tfrs_fa, axis=0)
                tfr_fa = tfr_fa.set_index(['subj'], append=True)
                tfr_miss = pd.concat(tfrs_miss, axis=0)
                tfr_miss = tfr_miss.set_index(['subj'], append=True)
                tfr_cr = pd.concat(tfrs_cr, axis=0)
                tfr_cr = tfr_cr.set_index(['subj'], append=True)
                
                hit = tfr_hit.loc[(tfr_hit.index.get_level_values('freq')>=f[0]) & (tfr_hit.index.get_level_values('freq')<=f[1]),:].groupby(['subj']).mean()
                fa = tfr_fa.loc[(tfr_fa.index.get_level_values('freq')>=f[0]) & (tfr_fa.index.get_level_values('freq')<=f[1]),:].groupby(['subj']).mean()
                miss = tfr_miss.loc[(tfr_miss.index.get_level_values('freq')>=f[0]) & (tfr_miss.index.get_level_values('freq')<=f[1]),:].groupby(['subj']).mean()
                cr = tfr_cr.loc[(tfr_cr.index.get_level_values('freq')>=f[0]) & (tfr_cr.index.get_level_values('freq')<=f[1]),:].groupby(['subj']).mean()

                ax = fig.add_subplot(2,2,plot_nr)

                if tl == 'stim':
                    time_ind = (hit.columns >= -0.3) & (hit.columns <= 1.1)
                elif tl == 'resp':
                    time_ind = (hit.columns >= -0.4) & (hit.columns <= 0.1)
                ax.plot(hit.loc[:,time_ind].mean(axis=0), color='orange')
                ax.plot(fa.loc[:,time_ind].mean(axis=0), ls='--', color='orange')
                ax.plot(miss.loc[:,time_ind].mean(axis=0), ls='--', color='green')
                ax.plot(cr.loc[:,time_ind].mean(axis=0), color='green')

                ax.set_title('Noise = {}'.format(n))
                ax.set_ylim(-10,10)
                plot_nr += 1

        shell()

        for sens in ['pos', 'lat',]:
        # for sens in ['pos']:
        # for sens in []:
            for n in [1,4]:
            # for n in [4]:
            # for n in [0,]:
                
                if sens == 'pos':
                    # contrasts = ['all', 'stimulus', 'choice_a', 'pupil', 'confidence', 'hit', 'fa', 'miss', 'cr']
                    contrasts = ['choice_a']
                else:
                    contrasts = ['hand',]
                
                for c in contrasts:
                    # fig = plt.figure(figsize=(4,2))
                    # gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1.25, 1])
                    
                    ratio = (0.1--0.7) / (1.1--0.30)
                    nr_clusters = 1
                    fig = plt.figure(figsize=(2.5,2)) 
                    gs = matplotlib.gridspec.GridSpec(1, nr_clusters*2, width_ratios=list(np.tile([1,ratio], nr_clusters)))
                                        
                    plot_nr_stim = 1
                    plot_nr_resp = 2
                    for a, tl in enumerate(['stim', 'resp',]):
                        tfr_group_a = []
                        tfr_group_b = []
                        for i, subj in enumerate(subjects):
                            print(subj)
                            trial_types = ['hit', 'fa', 'miss', 'cr', 'right', 'left', 'all', 'pupil_h', 'pupil_l', 'conf_h', 'conf_l']
                            # trial_types = ['hit', 'fa', 'miss', 'cr', 'right', 'left', 'all', 'pupil_h', 'pupil_l',]
                            
                            if len(which_sessions[subj]) > 0:
                            
                                if sens == 'lat':
                                    tfrs = []
                                    for tt in trial_types:
                                        tfr_session = []
                                        for session in which_sessions[subj]:
                                            # try:
                                            tfr_session.append( pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format('left', tl, subj, n, tt, session)), 'tfr') - \
                                                                    pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format('right', tl, subj, n, tt, session)), 'tfr') )
                                            # except:
                                            #     pass
                                        tfrs.append( pd.concat(tfr_session).groupby('freq').mean() )
                                else:
                                    tfrs = []
                                    for tt in trial_types:
                                        tfr_session = []
                                        for session in which_sessions[subj]:
                                            # try:
                                            tfr_session.append( pd.read_hdf(os.path.join(data_folder, 'sensor_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format(sens, tl, subj, n, tt, session)), 'tfr') )
                                            # except:
                                                # pass
                                        tfrs.append( pd.concat(tfr_session).groupby('freq').mean() )
                                    
                                if c == 'all':
                                    # tfr = (tfrs[0]+tfrs[1]+tfrs[2]+tfrs[3]) / 4.0
                                    tfr = tfrs[6]
                                if c == 'hit':
                                    tfr = tfrs[0]
                                if c == 'fa':
                                    tfr = tfrs[1]
                                if c == 'miss':
                                    tfr = tfrs[2]
                                if c == 'cr':
                                    tfr = tfrs[3]
                                if c == 'stimulus':
                                    tfr = (tfrs[0]+tfrs[2]) - (tfrs[1]+tfrs[3])   
                                if c == 'choice_a':
                                    tfr = (tfrs[0]+tfrs[1]) - (tfrs[2]+tfrs[3])    
                                if c == 'hand':
                                    tfr = tfrs[4]-tfrs[5] 
                                if c == 'pupil':
                                    tfr = tfrs[7]-tfrs[8]
                                if c == 'confidence':
                                    tfr = tfrs[9]-tfrs[10]
                                tfr["subj"] = i
                                tfr_group_a.append( tfr )
                        tfr = pd.concat(tfr_group_a, axis=0)
                        tfr = tfr.set_index(['subj'], append=True)

                        if c == 'choice_a':

                            # collapse across freq:
                            mean_contrast = tfr.loc[(tfr.index.get_level_values('freq')>=50) & (tfr.index.get_level_values('freq')<=150),:].groupby(['subj']).mean()
                            
                            # collapse across time:
                            mean_contrast = mean_contrast.loc[:,(mean_contrast.columns>=0) & (mean_contrast.columns<=1)].mean(axis=1)
                            fas = np.array([min(false_alarms[i]) for i in false_alarms.keys() if len(false_alarms[i]) > 0])
                            print(sp.stats.ttest_rel(mean_contrast, np.zeros(len(mean_contrast))))
                            print(sp.stats.pearsonr(mean_contrast, fas))

                        # time:
                        if tl  == 'stim':
                            time_cutoff = (-0.30, 1.1)
                            xlabel = 'Time from stimulus (s)'
                            timelock = 'stimlock'
                        if tl  == 'resp':
                            time_cutoff = (-0.7, 0.1)
                            xlabel = 'Time from report (s)'
                            timelock = 'resplock'

                        # vmin vmax:
                        if c == 'all':
                            vmin, vmax = (-15, 15)
                        else:
                            vmin, vmax = (-10, 10)
                        
                        ax = fig.add_subplot(gs[a]) 

                        if (c == 'stimulus') or (c == 'choice_a'):
                            plot_tfr(tfr, time_cutoff, vmin, vmax, timelock, cluster_correct=cluster_correct, threshold=0.05, plot_colorbar=False, weigthed_mean=True, weights=false_alarms, ax=ax)
                        else:
                            plot_tfr(tfr, time_cutoff, vmin, vmax, timelock, cluster_correct=cluster_correct, threshold=0.05, plot_colorbar=False, weigthed_mean=False, weights=None, ax=ax)
                        
                        if tl  == 'stim':
                            ax.set_ylabel('Frequency (Hz)')
                            ax.set_title('{} contrast'.format(c))
                            ax.axvline(-0.25, ls=':', lw=0.75, color='black',)
                            ax.axvline(-0.15, ls=':', lw=0.75, color='black',)
                            if (sens == 'pos') & (c == 'all'):
                                ax.add_patch(matplotlib.patches.Rectangle(xy=(0.3,70), width=0.7, height=30, lw=0.5, ls='--', color='w', fill=False))
                        if tl  == 'resp':
                            ax.set_title('N = {}'.format(len(np.unique(tfr.index.get_level_values('subj')))))
                            ax.get_yaxis().set_ticks([])
                            if (sens == 'lat') & (c == 'hand'):
                                ax.add_patch(matplotlib.patches.Rectangle(xy=(-0.5,12), width=0.5, height=24, lw=0.5, ls='--', color='w', fill=False))
                        ax.set_xlabel(xlabel)

                        if tl == 'stim':
                            fas = [min(false_alarms[i]) for i in false_alarms.keys() if len(false_alarms[i]) > 0]
                            fig2 = plt.figure(figsize=(12,12))
                            plt_nr = 1
                            for j, i in enumerate(np.unique(tfr.index.get_level_values('subj'))):
                                ax2 = fig2.add_subplot(5,4,plt_nr)
                                plot_tfr(tfr.loc[tfr.index.get_level_values('subj')==i], time_cutoff, -50, 50, timelock, cluster_correct=False, threshold=0.05, plot_colorbar=True, ax=ax2)
                                ax2.set_title("{}, {} FA's".format(subjects[i], fas[j]))
                                plt_nr += 1
                            fig2.tight_layout()
                            fig2.savefig(os.path.join(fig_folder, 'sensor_level', 'subjects', '{}_{}_{}_{}.png'.format(sens, n, c, timelock)))

                    gs.tight_layout(fig)
                    gs.update(wspace=0.05, hspace=0.05)
                    fig.savefig(os.path.join(fig_folder, 'sensor_level', '{}_{}_{}_{}.pdf'.format(sens, n, c, 0)))