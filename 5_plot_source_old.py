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

# data_folder = '/home/jw/share/data/'
# fig_folder = '/home/jw/share/figures/'
data_folder = 'Y:\\JW\\data'
fig_folder = 'Y:\\JW\\figures'
tfr_params = tfr_params = json.load(open('tfr_params.json'))
channels = list(pd.read_json("channels.json")[0])
foi = list(np.arange(4,162,2))

visual_field_clusters = {
     'vfcvisual':   (
                    u'lh.wang2015atlas.V1d-lh', u'rh.wang2015atlas.V1d-rh',
                    u'lh.wang2015atlas.V1v-lh', u'rh.wang2015atlas.V1v-rh',
                    u'lh.wang2015atlas.V2d-lh', u'rh.wang2015atlas.V2d-rh',
                    u'lh.wang2015atlas.V2v-lh', u'rh.wang2015atlas.V2v-rh',
                    u'lh.wang2015atlas.V3d-lh', u'rh.wang2015atlas.V3d-rh',
                    u'lh.wang2015atlas.V3v-lh', u'rh.wang2015atlas.V3v-rh',
                    u'lh.wang2015atlas.hV4-lh', u'rh.wang2015atlas.hV4-rh',
                    ),
     'vfcVO':       (
                    u'lh.wang2015atlas.VO1-lh', u'rh.wang2015atlas.VO1-rh', 
                    u'lh.wang2015atlas.VO2-lh', u'rh.wang2015atlas.VO2-rh',
                    ),
     'vfcPHC':      (
                    u'lh.wang2015atlas.PHC1-lh', u'rh.wang2015atlas.PHC1-rh',
                    u'lh.wang2015atlas.PHC2-lh', u'rh.wang2015atlas.PHC2-rh',
                    ),
     'vfcV3ab':     (
                    u'lh.wang2015atlas.V3A-lh', u'rh.wang2015atlas.V3A-rh', 
                    u'lh.wang2015atlas.V3B-lh', u'rh.wang2015atlas.V3B-rh',
                    ),
     'vfcTO':       (
                    u'lh.wang2015atlas.TO1-lh', u'rh.wang2015atlas.TO1-rh', 
                    u'lh.wang2015atlas.TO2-lh', u'rh.wang2015atlas.TO2-rh',
                    ),
     'vfcLO':       (
                    u'lh.wang2015atlas.LO1-lh', u'rh.wang2015atlas.LO1-rh', 
                    u'lh.wang2015atlas.LO2-lh', u'rh.wang2015atlas.LO2-rh',
                    ),
     'vfcIPS01':    (
                    u'lh.wang2015atlas.IPS0-lh', u'rh.wang2015atlas.IPS0-rh', 
                    u'lh.wang2015atlas.IPS1-lh', u'rh.wang2015atlas.IPS1-rh',
                    ),
     'vfcIPS2345':  (
                    u'lh.wang2015atlas.IPS2-lh', u'rh.wang2015atlas.IPS2-rh', 
                    u'lh.wang2015atlas.IPS3-lh', u'rh.wang2015atlas.IPS3-rh',
                    u'lh.wang2015atlas.IPS4-lh', u'rh.wang2015atlas.IPS4-rh', 
                    u'lh.wang2015atlas.IPS5-lh', u'rh.wang2015atlas.IPS5-rh',
                    ),
     'vfcSPL':      (
                    u'lh.wang2015atlas.SPL1-lh', u'rh.wang2015atlas.SPL1-rh',
                    ),
     'vfcFEF':      (
                    u'lh.wang2015atlas.FEF-lh', u'rh.wang2015atlas.FEF-rh',
                    ),
     }

glasser_clusters = {
     'HCPMMP1_visual_primary':                  ('lh.HCPMMP1_01_visual_primary-lh', 'rh.HCPMMP1_01_visual_primary-rh',),
     'HCPMMP1_visual_early':                    ('lh.HCPMMP1_02_visual_early-lh', 'rh.HCPMMP1_02_visual_early-rh',),
     'HCPMMP1_visual_dors':                     ('lh.HCPMMP1_03_visual_dors-lh', 'rh.HCPMMP1_03_visual_dors-rh',),
     'HCPMMP1_visual_ventral':                  ('lh.HCPMMP1_04_visual_ventral-lh', 'rh.HCPMMP1_04_visual_ventral-rh',),
     'HCPMMP1_visual_lateral':                  ('lh.HCPMMP1_05_visual_lateral-lh', 'rh.HCPMMP1_05_visual_lateral-rh',),
     'HCPMMP1_somatosensory_motor':             ('lh.HCPMMP1_06_somatosensory_motor-lh', 'rh.HCPMMP1_06_somatosensory_motor-rh',),
     'HCPMMP1_paracentral_lob_mid_cingulate':   ('lh.HCPMMP1_07_paracentral_lob_mid_cingulate-lh', 'rh.HCPMMP1_07_paracentral_lob_mid_cingulate-rh',),
     'HCPMMP1_premotor':                        ('lh.HCPMMP1_08_premotor-lh', 'rh.HCPMMP1_08_premotor-rh',),
     'HCPMMP1_opercular_posterior':             ('lh.HCPMMP1_09_opercular_posterior-lh', 'rh.HCPMMP1_09_opercular_posterior-rh',),
     'HCPMMP1_auditory_primary':                ('lh.HCPMMP1_10_auditory_primary-lh', 'rh.HCPMMP1_10_auditory_primary-rh',),
     'HCPMMP1_auditory_association':            ('lh.HCPMMP1_11_auditory_association-lh', 'rh.HCPMMP1_11_auditory_association-rh',),
     'HCPMMP1_insular_frontal_opercular':       ('lh.HCPMMP1_12_insular_frontal_opercular-lh', 'rh.HCPMMP1_12_insular_frontal_opercular-rh',),
     'HCPMMP1_temporal_medial':                 ('lh.HCPMMP1_13_temporal_medial-lh', 'rh.HCPMMP1_13_temporal_medial-rh',),
     'HCPMMP1_lateral_temporal':                ('lh.HCPMMP1_14_lateral_temporal-lh', 'rh.HCPMMP1_14_lateral_temporal-rh',),
     'HCPMMP1_temp_par_occ_junc':               ('lh.HCPMMP1_15_temporal_parietal_occipital_junction-lh', 'rh.HCPMMP1_15_temporal_parietal_occipital_junction-rh',),
     'HCPMMP1_parietal_superior':               ('lh.HCPMMP1_16_parietal_superior-lh', 'rh.HCPMMP1_16_parietal_superior-rh',),
     'HCPMMP1_parietal_inferior':               ('lh.HCPMMP1_17_parietal_inferior-lh', 'rh.HCPMMP1_17_parietal_inferior-rh',),
     'HCPMMP1_cingulate_posterior':             ('lh.HCPMMP1_18_cingulate_posterior-lh', 'rh.HCPMMP1_18_cingulate_posterior-rh',),
     'HCPMMP1_cingulate_anterior':              ('lh.HCPMMP1_19_cingulate_anterior_prefrontal_medial-lh', 'rh.HCPMMP1_19_cingulate_anterior_prefrontal_medial-rh',),
     'HCPMMP1_frontal_orbital_polar':           ('lh.HCPMMP1_20_frontal_orbital_polar-lh', 'rh.HCPMMP1_20_frontal_orbital_polar-rh',),
     'HCPMMP1_frontal_inferior':                ('lh.HCPMMP1_21_frontal_inferior-lh', 'rh.HCPMMP1_21_frontal_inferior-rh',),
     'HCPMMP1_prefrontal_dorsolateral':         ('lh.HCPMMP1_22_prefrontal_dorsolateral-lh', 'rh.HCPMMP1_22_prefrontal_dorsolateral-rh',),
}

jwg_clusters = {
     'JWG_aIPS':                                ('lh.JWG_lat_aIPS-lh', 'rh.JWG_lat_aIPS-rh',),
     'JWG_IPS_PCeS':                            ('lh.JWG_lat_IPS_PCeS-lh', 'rh.JWG_lat_IPS_PCeS-rh',),
     'JWG_M1':                                  ('lh.JWG_lat_M1-lh', 'rh.JWG_lat_M1-rh',),
}

all_clusters = {**visual_field_clusters, **glasser_clusters, **jwg_clusters}
areas = [item for sublist in [all_clusters[k] for k in all_clusters.keys()] for item in sublist]
print(areas)

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

def make_tfr_contrasts(subj, areas):
    
    # filenames:
    meta_filename_stim_a = os.path.join(data_folder, "epochs", subj, 'A', '{}-meta.hdf'.format('stimlock'))
    meta_filename_resp_a = os.path.join(data_folder, "epochs", subj, 'A', '{}-meta.hdf'.format('resplock'))
    meta_filename_stim_b = os.path.join(data_folder, "epochs", subj, 'B', '{}-meta.hdf'.format('stimlock'))
    meta_filename_resp_b = os.path.join(data_folder, "epochs", subj, 'B', '{}-meta.hdf'.format('resplock'))        

    tfr_filename_stim_l_a = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, 'A', 'stimlock', 'LF')))
    tfr_filename_resp_l_a = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, 'A', 'resplock', 'LF')))
    tfr_filename_stim_l_b = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, 'B', 'stimlock', 'LF')))
    tfr_filename_resp_l_b = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, 'B', 'resplock', 'LF')))
    tfr_filename_stim_h_a = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, 'A', 'stimlock', 'HF')))
    tfr_filename_resp_h_a = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, 'A', 'resplock', 'HF')))
    tfr_filename_stim_h_b = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, 'B', 'stimlock', 'HF')))
    tfr_filename_resp_h_b = glob.glob(os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_{}*-source.hdf".format(subj, 'B', 'resplock', 'HF')))

    # check if session exists:
    session_a = os.path.isfile(meta_filename_stim_a)
    session_b = os.path.isfile(meta_filename_stim_b)

    # load tfr and meta data:
    if session_a & session_b:
        meta_data_stim = pd.concat((pymeg.preprocessing.load_meta([meta_filename_stim_a])[0],
                                    pymeg.preprocessing.load_meta([meta_filename_stim_b])[0]), axis=0)

        meta_data_resp = pd.concat((pymeg.preprocessing.load_meta([meta_filename_resp_a])[0],
                                    pymeg.preprocessing.load_meta([meta_filename_resp_b])[0]), axis=0)
        tfr_data_stim_l = pd.concat((pd.concat([pd.read_hdf(f) for f in tfr_filename_stim_l_a], axis=0),
                                    pd.concat([pd.read_hdf(f) for f in tfr_filename_stim_l_b], axis=0)), axis=0)
        tfr_data_stim_h = pd.concat((pd.concat([pd.read_hdf(f) for f in tfr_filename_stim_h_a], axis=0),
                                    pd.concat([pd.read_hdf(f) for f in tfr_filename_stim_h_b], axis=0)), axis=0)
        tfr_data_resp_l = pd.concat((pd.concat([pd.read_hdf(f) for f in tfr_filename_resp_l_a], axis=0),
                                    pd.concat([pd.read_hdf(f) for f in tfr_filename_resp_l_b], axis=0)), axis=0)
        tfr_data_resp_h = pd.concat((pd.concat([pd.read_hdf(f) for f in tfr_filename_resp_h_a], axis=0),
                                    pd.concat([pd.read_hdf(f) for f in tfr_filename_resp_h_b], axis=0)), axis=0)

    elif session_a:
        meta_data_stim = pymeg.preprocessing.load_meta([meta_filename_stim_a])[0]
        meta_data_resp = pymeg.preprocessing.load_meta([meta_filename_resp_a])[0]
        tfr_data_stim_l = pd.concat([pd.read_hdf(f) for f in tfr_filename_stim_l_a], axis=0)
        tfr_data_stim_h = pd.concat([pd.read_hdf(f) for f in tfr_filename_stim_h_a], axis=0)
        tfr_data_resp_l = pd.concat([pd.read_hdf(f) for f in tfr_filename_resp_l_a], axis=0)
        tfr_data_resp_h = pd.concat([pd.read_hdf(f) for f in tfr_filename_resp_h_a], axis=0)
    elif session_b:
        meta_data_stim = pymeg.preprocessing.load_meta([meta_filename_stim_b])[0]
        meta_data_resp = pymeg.preprocessing.load_meta([meta_filename_resp_b])[0]
        tfr_data_stim_l = pd.concat([pd.read_hdf(f) for f in tfr_filename_stim_l_b], axis=0)
        tfr_data_stim_h = pd.concat([pd.read_hdf(f) for f in tfr_filename_stim_h_b], axis=0)
        tfr_data_resp_l = pd.concat([pd.read_hdf(f) for f in tfr_filename_resp_l_b], axis=0)
        tfr_data_resp_h = pd.concat([pd.read_hdf(f) for f in tfr_filename_resp_h_b], axis=0)

    # combine high and low freqs:
    tfr_data_stim = pd.concat((tfr_data_stim_l, tfr_data_stim_h))
    tfr_data_resp = pd.concat((tfr_data_resp_l, tfr_data_resp_h))

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

    # convert TFR data to pivot table:
    rois = tfr_data_stim.columns
    tfr_data_stim = pd.pivot_table(tfr_data_stim.reset_index(), values=rois, index=['trial', 'est_val'], columns='time').stack(-2)
    tfr_data_stim.index.names = ['trial', 'freq', 'channel']
    tfr_data_resp = pd.pivot_table(tfr_data_resp.reset_index(), values=rois, index=['trial', 'est_val'], columns='time').stack(-2)
    tfr_data_resp.index.names = ['trial', 'freq', 'channel']

    # collapse data across trial types:
    for area in rois:
        print(area)

        for n in [0,1,4]:
        # for n in [0]:
            for c in ['all', 'hit', 'fa', 'miss', 'cr', 'left', 'right', 'pupil_h', 'pupil_l']:
                
                # get condition indices:
                if n == 0:
                    trial_ind_stim = meta_data_stim.loc[meta_data_stim[c]==1, "hash"]
                    trial_ind_resp = meta_data_resp.loc[meta_data_stim[c]==1, "hash"]
                else:
                    trial_ind_stim = meta_data_stim.loc[(meta_data_stim[c]==1) & (meta_data_stim["noise"]==n), "hash"]
                    trial_ind_resp = meta_data_resp.loc[(meta_data_stim[c]==1) & (meta_data_stim["noise"]==n), "hash"]
                
                if len(trial_ind_stim) > 0:
                    
                    # apply condition ind, and collapse across trials:
                    tfr_data_stim_condition = tfr_data_stim.loc[tfr_data_stim.index.isin(trial_ind_stim, level='trial') & \
                                                    tfr_data_stim.index.isin([area], level='channel'),:].groupby(['freq', 'channel']).mean()
                    tfr_data_resp_condition = tfr_data_resp.loc[tfr_data_resp.index.isin(trial_ind_resp, level='trial') & \
                                                    tfr_data_resp.index.isin([area], level='channel'),:].groupby(['freq', 'channel']).mean()
                    
                    # get baseline:
                    baseline = baseline_per_sensor_get(tfr_data_stim_condition, baseline=(-0.25, -0.15))
                    
                    # apply baseline, and collapse across sensors:
                    tfr_data_stim_condition = baseline_per_sensor_apply(tfr_data_stim_condition, baseline=baseline).groupby(['freq']).mean()
                    tfr_data_resp_condition = baseline_per_sensor_apply(tfr_data_resp_condition, baseline=baseline).groupby(['freq']).mean()
                    
                    # save:
                    tfr_data_stim_condition.to_hdf(os.path.join(data_folder, 'source_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format(area, 'stim', subj, n, c, 'A')), 'tfr')
                    tfr_data_resp_condition.to_hdf(os.path.join(data_folder, 'source_level', 'contrasts', '{}_{}_{}_{}_{}_{}.hdf'.format(area, 'resp', subj, n, c, 'A')), 'tfr')
                    
                # # plot:
                # times = np.array(tfr_data_stim_condition.columns, dtype=float)
                # freqs = np.array(np.unique(tfr_data_stim_condition.index.get_level_values('freq')), dtype=float)
                # X = np.array(tfr_data_stim_condition)
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # cax = ax.pcolormesh(times, freqs, X, vmin=-15, vmax=15, cmap='jet')
                # fig.savefig('test.pdf')

if __name__ == '__main__':
    subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    # subjects = ['jw14', 'jw15']

    make_contrasts = False
    do_plots = True

    if make_contrasts:
        n_jobs = 6
        _ = Parallel(n_jobs=n_jobs)(delayed(make_tfr_contrasts)(subj, areas) for subj in subjects)
        
        # # serial:
        # for subj in subjects:
        #     print(subj)
        #     make_tfr_contrasts(subj, areas)

    if do_plots:
        # all_clusters = {'HCPMMP1_premotor': ('lh.HCPMMP1_08_premotor-lh', 'rh.HCPMMP1_08_premotor-rh',),}
        for cluster in all_clusters.keys():
        # for cluster in visual_field_clusters.keys():
        # for cluster in glasser_clusters.keys():
        # for cluster in jwg_clusters.keys():
            # for n in [0,1,4]:
            for n in [0,1,4]:
                # contrasts = ['all', 'hand', 'stimulus', 'choice_a', 'pupil',]
                contrasts = ['all', 'hand', 'choice_a', 'pupil']
                for c in contrasts:

                    if c == 'hand':
                        lat = True
                    else:
                        lat = False
                        
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