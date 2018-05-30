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
channels = list(pd.read_json("channels.json")[0])

# TFR parameters:
foi_l = np.arange(4,41,1)
params_l = {
            'foi': foi_l, # frequencies of interest
            'cycles': foi_l * 0.4, # foi * window_length
            'n_tapers': 1 + 1,  # desired tapers + 1
            'decim': 10, # downsample resuling TFR
            }
foi_h = np.arange(42,162,2)
params_h = {
            'foi': foi_h, # frequencies of interest
            'cycles': foi_h * 0.4, # foi * window_length
            'n_tapers': 5 + 1,  # desired tapers + 1
            'decim': 10, # downsample resuling TFR
            }

# describe taper:
pymeg.tfr.describe_taper(params_l["foi"], params_l["cycles"], params_l["n_tapers"])
pymeg.tfr.describe_taper(params_h["foi"], params_h["cycles"], params_h["n_tapers"])
# pymeg.tfr.tiling_plot(**tfr_params)

# subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
# subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13']
subjects = ['jw22', 'jw23', 'jw24', 'jw30']

# tfr:
for subj in subjects:
    for session in ["A", "B"]:
        for tl in ['stimlock', 'resplock']:
            
            filename = os.path.join(data_folder, "epochs", subj, session, '{}-epo.fif.gz'.format(tl))
            if os.path.isfile(filename):
                
                try:
                
                	# compute tfr:
                	power_l = pymeg.tfr.tfr(filename=filename, method='multitaper', foi=params_l["foi"], 
                                            cycles=params_l["cycles"], time_bandwidth=params_l["n_tapers"], 
                                            decim=params_l["decim"], n_jobs=2, save=False)
                	power_h = pymeg.tfr.tfr(filename=filename, method='multitaper', foi=params_h["foi"], 
                                            cycles=params_h["cycles"], time_bandwidth=params_h["n_tapers"], 
                                            decim=params_h["decim"], n_jobs=2, save=False)
                
                	# new object:
                	power = power_l.copy()
                	power.freqs = np.concatenate((power_l.freqs, power_h.freqs))
                	power.data = np.dstack((power_l.data, power_h.data))
                	
                	# delete from mem:
                	del power_l
                	del power_h

                	# save:
                	epochs = mne.read_epochs(filename)
                	outname = filename.replace('epo.fif.gz', 'tfr.hdf')
                	
                	try:
                		os.remove(outname)
                	except:
                		pass

                	pymeg.tfr.save_tfr(power, outname, epochs.events)
                	
                	# delete from mem:
                	del power

                except Exception as e:

                	print(e)

                	shell()


                
                
                