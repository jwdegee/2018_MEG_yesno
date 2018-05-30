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
from pymeg import preprocessing
from pymeg import tfr

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

# data_folder = '/home/jwdegee/degee/MEG/data/'
data_folder = '/home/jw/share/data/'
tfr_params = json.load(open('tfr_params.json'))
channels = list(pd.read_json("channels.json")[0])

# subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
# subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14',]
subjects = ['jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']

def get_meta(raw, mapping, run_nr):
    meta, timing = pymeg.preprocessing.get_meta(raw, mapping, {}, 8, 8)
    for c in meta:
        if c in [v[0] for v in mapping.values()] or str(c).endswith('time'):
            continue
        del meta[c]
    meta.loc[:, 'hash'] = timing.baseline_start_time.values + (10000000 * (run_nr + 1))
    meta.loc[:, 'block'] = meta.index.values/102
    #timing = timing.set_index('hash')
    timing.loc[:, 'block'] = meta.block
    return meta, timing

mapping = {30: ('wait_fix', 0),
        8: ('baseline_start', 0),
        16: ('stim_meg', -2),
        17: ('stim_meg', -1),
        18: ('stim_meg', 1),
        19: ('stim_meg', 2),
        32: ('resp_meg', -2),
        33: ('resp_meg', -1),
        34: ('resp_meg', 1),
        35: ('resp_meg', 2),
        64: ('conf_meg', -2),
        65: ('conf_meg', -1),
        66: ('conf_meg', 1),
        67: ('conf_meg', 2),
        }

# preprocessing & epochs:
for subj in subjects:
    for session in ["A", "B"]:
        
        epochs_stimlock = []
        epochs_resplock = []
        metas_stimlock = []
        metas_resplock = []
        
        runs = sorted([run.split('/')[-1] for run in glob.glob(os.path.join(data_folder, "raw", subj, session, "meg", "*.ds"))])
        this_run = 0
        
        for run in runs:
            
            # load raw MEG:
            filename = os.path.join(data_folder, "raw", subj, session, "meg", run)
            raw = mne.io.read_raw_ctf(filename)
            mb, tb = get_meta(raw, mapping, this_run)
            
            # load blink data, and attach to MEG meta data:
            filename = glob.glob(os.path.join(data_folder, "pupil", "{}_{}_pupil_data.csv".format(subj, session)))[0]
            df_pupil = pd.read_csv(filename).drop(columns=["Unnamed: 0"])
            df_pupil = df_pupil.loc[df_pupil["run_nr"]==this_run,:].reset_index()
            df_pupil['rt'] = df_pupil['rt']
        
            # throw away some weird trials (nan for confidence etc), and throw away blink + saccade trials:
            weird = np.array(pd.isnull(mb["resp_meg"]) | pd.isnull(mb["conf_meg"])  | pd.isnull(tb["resp_meg_time"])  | pd.isnull(tb["conf_meg_time"])) \
                     | np.array([isinstance(i,list) for i in mb["resp_meg"]]) | np.array([isinstance(i,list) for i in tb["resp_meg_time"]]) \
                     | np.array([isinstance(i,list) for i in mb["conf_meg"]]) | np.array([isinstance(i,list) for i in tb["conf_meg_time"]])
            bs = np.array(df_pupil["blinks_nr"]>0)
            omit = weird | bs
            mb = mb.loc[~omit, :]
            tb = tb.loc[~omit, :]
            df_pupil = df_pupil.loc[~omit, :]
            print("delete {} weird trials".format(sum(weird)))
            print("delete {} blink/sac trials".format(sum(bs)))
            print("delete {} total trials".format(sum(omit)))
        
            # plot fraction of trials thrown away:
            if not os.path.exists(os.path.join(data_folder, "epochs", subj, session, "plots")):
                os.makedirs(os.path.join(data_folder, "epochs", subj, session, "plots"))
            labels = ['weird', 'blinks', 'good']
            sizes = [sum(weird), sum(bs), df_pupil.shape[0]]
            colors = ['lightcoral', 'coral', 'green']
            explode = (0, 0, 0.1)
            fig, ax = plt.subplots(figsize=(2,2))
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')
            fig.savefig(os.path.join(data_folder, "epochs", subj, session, "plots", "trials_{}.pdf".format(this_run)))
        
            # change response times to button box rather than trigger:
            # trigger_ts = raw.get_data(2).ravel()
            # button_ts = raw.get_data(3).ravel()
            
            buttons = pd.DataFrame(mne.find_events(raw, 'UPPT002', shortest_event=1), columns=['time', 'dummy', 'id'])
            response_times = np.zeros(tb.shape[0])
            confidence_times = np.zeros(tb.shape[0])
            for i in range(tb.shape[0]):
                response_times[i] = int(buttons.loc[(buttons['time']-tb.iloc[i]["resp_meg_time"]).abs().argsort()[0],"time"])
                confidence_times[i] = int(buttons.loc[(buttons['time']-tb.iloc[i]["conf_meg_time"]).abs().argsort()[0],"time"])
            ind = (tb["resp_meg_time"] - response_times) < 100
            tb.loc[ind,"resp_meg_time"] = response_times[ind]
            ind = (tb["conf_meg_time"] - confidence_times) < 100
            tb.loc[ind,"conf_meg_time"] = confidence_times[ind]
            tb['rt_meg'] = ((tb['resp_meg_time'] - tb['stim_meg_time']) / raw.info["sfreq"])
            
            # plot rt differences:
            if not os.path.exists(os.path.join(data_folder, "epochs", subj, session, "plots")):
                os.makedirs(os.path.join(data_folder, "epochs", subj, session, "plots"))
            fig, ax = plt.subplots(figsize=(2,2))
            ax.hist(np.array(df_pupil['rt'])-tb['rt_meg'], bins=6)
            plt.xlabel('RT diff (s)')
            plt.ylabel('Trials')
            sns.despine(offset=5, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(data_folder, "epochs", subj, session, "plots", "rt_diff_{}.pdf".format(this_run)))
            
            # concat meta:
            mb = pd.concat((mb, df_pupil), axis=1)
            
            # artefact rejection:
            r, ants, artdef = pymeg.preprocessing.preprocess_block(raw, blinks=False)
            
            # notch filter:
            r.load_data()
            midx = np.where([x.startswith('M') for x in r.ch_names])[0]
            freqs = np.arange(50, 251, 50)
            notch_widths = freqs / 100.0
            r.notch_filter(freqs=freqs, notch_widths=notch_widths, picks=midx)
            
            # epochs:
            stimlock_meta, stimlock = pymeg.preprocessing.get_epoch(r, mb, tb, event='stim_meg_time', 
                                                                    epoch_time=(-0.5, 1.5), epoch_label='hash')
            stimlock_meta = stimlock_meta.T.drop_duplicates().T
            resplock_meta, resplock = pymeg.preprocessing.get_epoch(r, mb, tb, event='resp_meg_time',
                                                                    epoch_time=(-0.9, 0.4), epoch_label='hash')
            resplock_meta = resplock_meta.T.drop_duplicates().T
            
            # downsample epochs
            stimlock.resample(600, npad="auto")
            resplock.resample(600, npad="auto")
            
            # # save:
            # if not os.path.exists(os.path.join(data_folder, "epochs", subj, session)):
            #     os.makedirs(os.path.join(data_folder, "epochs", subj, session))
            # stimlock_meta.to_hdf(os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_stimlock-meta.hdf'), 'meta')
            # stimlock.save(os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_stimlock-epo.fif.gz'))
            # resplock_meta.to_hdf(os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_resplock-meta.hdf'), 'meta')
            # resplock.save(os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_resplock-epo.fif.gz'))
            
            # # # load:
            # # stimlock = mne.read_epochs(os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_stimlock-epo.fif.gz'))
            # # stimlock_meta = pd.read_hdf(os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_stimlock-meta.hdf'))
            
            # append:
            epochs_stimlock.append(stimlock)
            epochs_resplock.append(resplock)
            metas_stimlock.append(stimlock_meta)
            metas_resplock.append(resplock_meta)
            
            # update run:
            this_run += 1
        
        # across runs:
        if len(runs) > 0:
            
            # set head position to middle run:
            center = int(np.floor(len(runs) / 2.0))
            for r in range(len(runs)):
                epochs_stimlock[r].info.update({'dev_head_t':epochs_stimlock[center].info.get('dev_head_t')})
                epochs_resplock[r].info.update({'dev_head_t':epochs_resplock[center].info.get('dev_head_t')})
            
            # concatenate:
            epochs_stimlock = mne.concatenate_epochs(epochs_stimlock)
            epochs_resplock = mne.concatenate_epochs(epochs_resplock)
            metas_stimlock = pd.concat(metas_stimlock, axis=0)
            metas_resplock = pd.concat(metas_resplock, axis=0)
            
            # save:
            epochs_stimlock.save(os.path.join(data_folder, "epochs", subj, session, 'stimlock-epo.fif.gz'))
            epochs_resplock.save(os.path.join(data_folder, "epochs", subj, session, 'resplock-epo.fif.gz'))
            metas_stimlock.to_hdf(os.path.join(data_folder, "epochs", subj, session, 'stimlock-meta.hdf'), 'meta')
            metas_resplock.to_hdf(os.path.join(data_folder, "epochs", subj, session, 'resplock-meta.hdf'), 'meta')
            
            
            # if do_tfr:
            #
            #     # params:
            #     # foi = tfr_params["foi"]
            #     foi = np.arange(4,162,2)
            #     cycles = np.array(foi) * tfr_params["window_length"]
            #     n_tapers = tfr_params["n_tapers"]+1
            #     decim = tfr_params["decim"]
            #
            #     # describe taper:
            #     pymeg.tfr.describe_taper(foi, cycles, n_tapers)
            #     # pymeg.tfr.tiling_plot(**tfr_params)
            #
            #     # tfr:
            #     pymeg.tfr.tfr(filename=os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_stimlock-epo.fif.gz'),
            #                     method='multitaper', foi=foi, cycles=cycles, time_bandwidth=n_tapers, decim=decim, n_jobs=1,)
            #     pymeg.tfr.tfr(filename=os.path.join(data_folder, "epochs", subj, session, run.split('.')[0] + '_resplock-epo.fif.gz'),
            #                     method='multitaper', foi=foi, cycles=cycles, time_bandwidth=n_tapers, decim=decim, n_jobs=1,)
            
            