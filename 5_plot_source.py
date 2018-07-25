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
from joblib import Memory

from IPython import embed as shell

import pymeg
from pymeg import preprocessing as prep
from pymeg import tfr as tfr
from pymeg import atlas_glasser

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

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)

# data_folder = '/home/jw/share/data/'
# fig_folder = '/home/jw/share/figures/'
data_folder = 'Z:\\JW\\data'
fig_folder = 'Z:\\JW\\figures'
# data_folder = '/home/jwdegee/degee/MEG/data/'
# fig_folder = '/home/jwdegee/degee/MEG/figures/'
tfr_params = tfr_params = json.load(open('tfr_params.json'))
channels = list(pd.read_json("channels.json")[0])
foi = list(np.arange(4,162,2))

def load_tfr_data(subj, session, timelock):
    
    tfr_data_lf_filenames = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, session, timelock, 'LF')))
    tfr_data_hf_filenames = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, session, timelock, 'HF')))
    
    print(tfr_data_lf_filenames)
    print(tfr_data_hf_filenames)

    tfr_data = pd.concat((pd.concat([pd.read_hdf(f) for f in tfr_data_lf_filenames], axis=0), 
                                pd.concat([pd.read_hdf(f) for f in tfr_data_hf_filenames], axis=0)))
    print('load complete')

    # convert TFR data to pivot table:
    tfr_data = pd.pivot_table(tfr_data.reset_index(), values=tfr_data.columns, index=['trial', 'est_val'], columns='time').stack(-2)
    tfr_data.index.names = ['trial', 'freq', 'channel']
    print('pivot complete')

    return tfr_data

def load_meta_data(subj, session, timelock):

    meta_data_filename = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format(timelock))
    meta_data = pymeg.preprocessing.load_meta([meta_data_filename])[0]

    # fix meta data:
    meta_data["all"] = 1
    meta_data["left"] = (meta_data["resp_meg"] < 0).astype(int)
    meta_data["right"] = (meta_data["resp_meg"] > 0).astype(int)
    meta_data["hit"] = ((meta_data["stimulus"] == 1) & (meta_data["choice_a"] == 1)).astype(int)
    meta_data["fa"] = ((meta_data["stimulus"] == 0) & (meta_data["choice_a"] == 1)).astype(int)
    meta_data["miss"] = ((meta_data["stimulus"] == 1) & (meta_data["choice_a"] == 0)).astype(int)
    meta_data["cr"] = ((meta_data["stimulus"] == 0) & (meta_data["choice_a"] == 0)).astype(int)
    meta_data["left"] = (meta_data["resp_meg"] < 0).astype(int)
    meta_data["right"] = (meta_data["resp_meg"] > 0).astype(int)
    meta_data["pupil_h"] = (meta_data["pupil_lp_d"] >= np.percentile(meta_data["pupil_lp_d"], 60)).astype(int)
    meta_data["pupil_l"] = (meta_data["pupil_lp_d"] <= np.percentile(meta_data["pupil_lp_d"], 40)).astype(int)

    return meta_data

def baseline_per_sensor_get(tfr, baseline=(-0.4, 0)):
    '''
    Get average baseline
    '''
    time = tfr.columns.get_level_values('time').values.astype(float)
    id_base =  (time > baseline[0]) & (time < baseline[1])
    base = tfr.loc[:, id_base].groupby(['freq', 'channel']).mean().mean(axis=1)  # This should be len(nr_freqs * nr_hannels)
    return base
   
def baseline_per_sensor_apply(tfr, baseline):
    '''
    Baseline correction by dividing by average baseline
    '''
    def div(x):
        bval = float(baseline.loc[baseline.index.isin([x.index.get_level_values('freq').values[0]], level='freq') & 
                                  baseline.index.isin([x.index.get_level_values('channel').values[0]], level='channel')])
        return (x - bval) / bval * 100
    return tfr.groupby(['freq', 'channel']).apply(div)

def make_tfr_contrasts(tfr_data, tfr_data_to_baseline, meta_data, area, condition):

    condition_ind = meta_data.loc[meta_data[condition]==1, "hash"]

    # apply condition ind, collapse across trials, and get baseline::
    tfr_data_to_baseline = tfr_data_to_baseline.loc[tfr_data_to_baseline.index.isin(condition_ind, level='trial') & \
                                    tfr_data_to_baseline.index.isin([area], level='channel'),:].groupby(['freq', 'channel']).mean()
    baseline = baseline_per_sensor_get(tfr_data_to_baseline, baseline=(-0.25, -0.15))
        
    # apply condition ind, and collapse across trials:
    tfr_data_condition = tfr_data.loc[tfr_data.index.isin(condition_ind, level='trial') & \
                                    tfr_data.index.isin([area], level='channel'),:].groupby(['freq', 'channel']).mean()

    # apply baseline, and collapse across sensors:
    tfr_data_condition = baseline_per_sensor_apply(tfr_data_condition, baseline=baseline).groupby(['freq']).mean()
        
    return tfr_data_condition

@memory.cache
def load_tfr_contrast(subj, sessions, timelock, areas, conditions):

    tfr_conditions = []

    for session in sessions:
        print(session)

        try: # some subjects have only one session...

            # load data:
            tfr_data = load_tfr_data(subj, session, timelock)
            meta_data = load_meta_data(subj, session, timelock)
            
            # data to baseline:
            if timelock == 'resplock':
                tfr_data_to_baseline = load_tfr_data(subj, session, 'stimlock')
            else:
                tfr_data_to_baseline = tfr_data
            
            # compute contrasts:
            for area in areas:
                for condition in conditions:

                    print(area)
                    print(condition)
                    tfr_data_condition = make_tfr_contrasts(tfr_data, tfr_data_to_baseline, meta_data, area, condition)
                    tfr_data_condition['subj'] = subj
                    tfr_data_condition['session'] = session
                    tfr_data_condition['area'] = area
                    tfr_data_condition['condition'] = condition
                    tfr_data_condition = tfr_data_condition.set_index(['subj', 'session', 'area', 'condition',], append=True, inplace=False)
                    tfr_data_condition = tfr_data_condition.reorder_levels(['subj', 'session', 'area', 'condition', 'freq'])
                    tfr_conditions.append(tfr_data_condition)
        except:
            pass
    tfr_condition = pd.concat(tfr_conditions)
    return tfr_condition

def plot_tfr(tfr, time_cutoff, vmin, vmax, tl, cluster_correct=False, threshold=0.05, ax=None):

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
        if tl  == 'stim':
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
    
    if tl == 'stim':
        ax.set_xlabel('Time from stimulus (s)')
        ax.axvline(-0.25, ls=':', lw=0.75, color='black',)
        ax.axvline(-0.15, ls=':', lw=0.75, color='black',)
        ax.set_ylabel('Frequency (Hz)')
        # ax.set_title('{} contrast'.format(c))
    elif tl == 'resp':
        ax.set_xlabel('Time from report (s)')
        # ax.set_title('N = {}'.format(len(subjects)))
        ax.tick_params(labelleft='off')
        plt.colorbar(cax, ticks=[vmin, 0, vmax])

    return ax

if __name__ == '__main__':
    
    # subjects:
    subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    subjects = ['jw01', 'jw02', 'jw03', 'jw05', ]
    
    # areas and clusters:
    all_clusters, visual_field_clusters, glasser_clusters, jwg_clusters = atlas_glasser.get_clusters()
    areas = [item for sublist in [all_clusters[k] for k in all_clusters.keys()] for item in sublist]

    # contrasts:
    contrasts = {
                'all' : ['all'], 
                'choice': ['hit', 'fa', 'miss', 'cr'],
                'hand' : ['left', 'right'],
                'pupil': ['pupil_h', 'pupil_l'],
                }
    conditions = [item for sublist in [contrasts[k] for k in contrasts.keys()] for item in sublist]
    conditions = ['all', 'hit', 'fa', 'miss', 'cr', 'left', 'right', 'pupil_h', 'pupil_l']
    
    # load for all subjects:    
    tfr_conditions_stim = []
    tfr_conditions_resp = []
    for subj in subjects:
        tfr_conditions_stim.append(load_tfr_contrast(subj, ['A','B'], 'stimlock', areas, conditions))
        tfr_conditions_resp.append(load_tfr_contrast(subj, ['A','B'], 'resplock', areas, conditions))
    tfr_conditions_stim = pd.concat(tfr_conditions_stim)
    tfr_conditions_resp = pd.concat(tfr_conditions_resp)

    # mean across sessions:
    tfr_conditions_stim = tfr_conditions_stim.groupby(['subj', 'area', 'condition', 'freq']).mean()
    tfr_conditions_resp = tfr_conditions_resp.groupby(['subj', 'area', 'condition', 'freq']).mean()
    
    # all_clusters = {'HCPMMP1_premotor': ('lh.HCPMMP1_08_premotor-lh', 'rh.HCPMMP1_08_premotor-rh',),}
    # for cluster in all_clusters.keys():
    # for cluster in visual_field_clusters.keys():
    for cluster in jwg_clusters.keys():
    # for cluster in glasser_clusters.keys():
    
        print(cluster)
        for c in contrasts.keys():
            print(c)
            if c == 'hand':
                lat = True
            else:
                lat = False
            
            fig = plt.figure(figsize=(4,2))
            gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1.25, 1])
            
            for i, tl in enumerate(['stim', 'resp',]):

                if tl == 'stim':
                    tfr_condition = tfr_conditions_stim.copy()
                elif tl == 'resp':
                    tfr_condition = tfr_conditions_resp.copy()

                if lat:
                    tfrs_rh = [pd.concat([tfr_condition.loc[(tfr_condition.index.isin([area], level='area')) &\
                                                            (tfr_condition.index.isin([condition], level='condition'))]\
                                        for area in all_clusters[cluster] if 'rh' in area]).groupby(['subj', 'freq']).mean()\
                                        for condition in contrasts[c]]                
                    tfrs_lh = [pd.concat([tfr_condition.loc[(tfr_condition.index.isin([area], level='area')) &\
                                                            (tfr_condition.index.isin([condition], level='condition'))]\
                                        for area in all_clusters[cluster] if 'lh' in area]).groupby(['subj', 'freq']).mean()\
                                        for condition in contrasts[c]]
                    tfrs = [tfrs_rh[i] - tfrs_lh[i] for i in range(len(tfrs_lh))]

                else:
                    tfrs = [pd.concat([tfr_condition.loc[(tfr_condition.index.isin([area], level='area')) &\
                                                         (tfr_condition.index.isin([condition], level='condition'))]\
                                        for area in all_clusters[cluster]]).groupby(['subj', 'freq']).mean()\
                                        for condition in contrasts[c]]

                # create contrasts:
                if c == 'all':
                    tfr = tfrs[0]
                if c == 'stimulus':
                    tfr = (tfrs[0]+tfrs[2]) - (tfrs[1]+tfrs[3])   
                if c == 'choice':
                    tfr = (tfrs[0]+tfrs[1]) - (tfrs[2]+tfrs[3])    
                if c == 'hand':
                    tfr = tfrs[0]-tfrs[1] 
                if c == 'pupil':
                    tfr = tfrs[0]-tfrs[1]

                # time:
                if tl  == 'stim':
                    time_cutoff = (-0.35, 1.3)
                    xlabel = 'Time from stimulus (s)'
                if tl  == 'resp':
                    time_cutoff = (-0.7, 0.2)
                    xlabel = 'Time from response (s)'

                # vmin vmax:
                if c == 'all':
                    vmin, vmax = (-15, 15)
                else:
                    vmin, vmax = (-10, 10)

                # plot:
                ax = fig.add_subplot(gs[i])
                plot_tfr(tfr, time_cutoff, vmin, vmax, tl=tl, cluster_correct=False, threshold=0.05, ax=ax)

            fig.tight_layout()
            fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_{}_{}.pdf'.format(cluster, 0, c, int(lat))))