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
# data_folder = 'Y:\\JW\\data'
# fig_folder = 'Y:\\JW\\figures'
data_folder = '/home/jwdegee/degee/MEG/data/'
fig_folder = '/home/jwdegee/degee/MEG/figures/'
tfr_params = tfr_params = json.load(open('tfr_params.json'))
channels = list(pd.read_json("channels.json")[0])
foi = list(np.arange(4,162,2))

def load_tfr_data(subj, session, timelock):
    
    tfr_data_lf_filenames = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, session, timelock, 'LF')))
    tfr_data_hf_filenames = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, session, timelock, 'HF')))
    
    print tfr_data_lf_filenames
    print tfr_data_hf_filenames

    tfr_data = pd.concat((pd.concat([pd.read_hdf(f) for f in tfr_data_lf_filenames], axis=0), 
                                pd.concat([pd.read_hdf(f) for f in tfr_data_hf_filenames], axis=0)))

    print('load complete')

    shell()

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

# @memory.cache
def make_tfr_contrasts(tfr_data, tfr_data_to_baseline, meta_data, area, condition):

    condition_ind = meta_data.loc[meta_data[condition]==1, "hash"]

    # apply condition ind, collapse across trials, and get baseline::
    tfr_data_to_baseline = tfr_data_to_baseline.loc[tfr_data_to_baseline.index.isin(condition_ind, level='trial') & \
                                    tfr_data_to_baseline.index.isin([area], level='channel'),:].groupby(['freq', 'channel']).mean()
    baseline = baseline_per_sensor_get(tfr_data_to_baseline, baseline=(-0.25, -0.15))
        
    # apply condition ind, and collapse across trials:
    tfr_data_condition = tfr_data.loc[tfr_data.index.isin(condition_ind, level='trial') & \
                                    tfr_data.index.isin([area], level='channel'),:].groupby(['freq', 'channel']).mean()
    print tfr_data_condition.head()

    # apply baseline, and collapse across sensors:
    tfr_data_condition = baseline_per_sensor_apply(tfr_data_condition, baseline=baseline).groupby(['freq']).mean()
        
    return tfr_data_condition

# @memory.cache
def load_tfr_contrast(subj, session, timelock, areas, conditions):

    tfr_data = load_tfr_data(subj, session, timelock)
    meta_data = load_meta_data(subj, session, timelock)
    
    shell()

    if timelock == 'resplock':
        tfr_data_to_baseline = load_tfr_data(subj, session, 'stimlock')
    else:
        tfr_data_to_baseline = tfr_data
    for area in areas:
        tfr_conditions = {}
        for condition in conditions:

            print(area)
            print(condition)

            tfr_data_condition = make_tfr_contrasts(tfr_data, tfr_data_to_baseline, meta_data, area, condition)
            # tfr_conditions{condition : tfr_data_condition}


if __name__ == '__main__':
    subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    
    all_clusters, visual_field_clusters, glasser_clusters, jwg_clusters = atlas_glasser.get_clusters()

    conditions = ['all', 'hit', 'fa', 'miss', 'cr', 'left', 'right', 'pupil_h', 'pupil_l']
    areas = [item for sublist in [all_clusters[k] for k in all_clusters.keys()] for item in sublist]
    for subj in subjects:
        load_tfr_contrast(subj, 'A', 'stimlock', areas, conditions)



    # all_clusters = {'HCPMMP1_premotor': ('lh.HCPMMP1_08_premotor-lh', 'rh.HCPMMP1_08_premotor-rh',),}
    for cluster in all_clusters.keys():
    # for cluster in visual_field_clusters.keys():
    # for cluster in glasser_clusters.keys():
    # for cluster in jwg_clusters.keys():
        # for n in [0,1,4]:
        for n in [0]:
            # contrasts = ['all', 'hand', 'stimulus', 'choice_a', 'pupil',]
            contrasts = ['all', 'hand', 'choice_a', 'pupil']
            for c in contrasts:
                if c == 'hand':
                    lat = True
                else:
                    lat = False
                




                


                shell()





                










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

                        tfrs = []
                        for tt in trial_types:
                            tfr_session = []
                            
                            if lat:
                                for area in all_clusters[cluster]:
                                    if 'lh.' in area:
                                        try:
                                            tfr_dum = pd.read_hdf(os.path.join(data_folder, 'source_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format(area, tl, subj, n, tt, 'A')), 'tfr') - \
                                                      pd.read_hdf(os.path.join(data_folder, 'source_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format(area.replace('lh', 'rh'), tl, subj, n, tt, 'A')), 'tfr')
                                            tfr_session.append(tfr_dum)
                                        except:
                                            pass
                            else:
                                for area in all_clusters[cluster]:
                                    try:
                                        tfr_session.append( pd.read_hdf(os.path.join(data_folder, 'source_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format(area, tl, subj, n, tt, 'A')), 'tfr') )
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
                        time_cutoff = (-0.35, 1.3)
                        xlabel = 'Time from stimulus (s)'
                    if tl  == 'resp':
                        time_cutoff = (-0.7, 0.2)
                        xlabel = 'Time from response (s)'
            
                    # vmin vmax:
                    if c == 'all':
                        vmin, vmax = (-15, 15)
                    elif 'cor' in c:
                        vmin, vmax = (-0.75, 0.75)
                    else:
                        vmin, vmax = (-10, 10)
            
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
                        
                        try:
                            T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(test_data, threshold={'start':0, 'step':0.2}, 
                                                                                                        connectivity=None, tail=0, n_permutations=1000, n_jobs=10)
                            sig = cluster_p_values.reshape((test_data.shape[1], test_data.shape[2]))
                            ax.contour(times_test_data, freqs, sig, (threshold,), linewidths=0.5, colors=('black'))
                        except:
                            pass

                    ax.axvline(0, ls='--', lw=0.75, color='black',)
                    ax.set_xlabel(xlabel)
                    if a == 0:
                        ax.axvline(-0.25, ls=':', lw=0.75, color='black',)
                        ax.axvline(-0.15, ls=':', lw=0.75, color='black',)
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
                fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_{}_{}.pdf'.format(cluster, n, c, int(lat))))
                
                fig_s.tight_layout()
                fig_s.savefig(os.path.join(fig_folder, 'source_level', 'subjects', '{}_{}_{}_{}.png'.format(cluster, n, c, int(lat))))