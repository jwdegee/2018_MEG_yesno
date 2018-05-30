import mne
import os

subjects_dir = '/home/jw/share/data/fs_subjects/'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)

# these should correspond to your FS subject IDs:
subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']

for subj in subjects:
    for hemi in ['lh', 'rh']:
        
        # transform atlas to individual space:
        cmd = 'mris_apply_reg --src-annot {} --trg {} --streg {} {}'.format(\
                            os.path.join(subjects_dir, 'fsaverage', 'label', '{}.HCPMMP1_combined.annot'.format(hemi)),
                            os.path.join(subjects_dir, subj, 'label', '{}.HCPMMP1_combined.annot'.format(hemi)),
                            os.path.join(subjects_dir, 'fsaverage', 'surf', '{}.sphere.reg'.format(hemi)),
                            os.path.join(subjects_dir, subj, 'surf', '{}.sphere.reg'.format(hemi)),)
        os.system(cmd)
        
        # unpack into labels:
        cmd = 'mri_annotation2label --subject {} --hemi {} --labelbase {} --annotation {}'.format(
                        subj, 
                        hemi, 
                        '{}.HCPMMP1_combined'.format(hemi),
                        'HCPMMP1_combined'.format(hemi),
                        )
        os.system(cmd)
        
        # rename in alphabetical order...
        orig_names = [
                    '_Inside', 
                    'Anterior Cingulate and Medial Prefrontal Cortex',
                    'Auditory Association Cortex', 
                    'Dorsal Stream Visual Cortex',
                    'DorsoLateral Prefrontal Cortex', 
                    'Early Auditory Cortex',
                    'Early Visual Cortex', 
                    'Inferior Frontal Cortex',
                    'Inferior Parietal Cortex', 
                    'Insular and Frontal Opercular Cortex',
                    'Lateral Temporal Cortex',
                    'MT+ Complex and Neighboring Visual Areas',
                    'Medial Temporal Cortex', 
                    'Orbital and Polar Frontal Cortex',
                    'Paracentral Lobular and Mid Cingulate Cortex',
                    'Posterior Cingulate Cortex', 
                    'Posterior Opercular Cortex',
                    'Premotor Cortex', 
                    'Primary Visual Cortex (V1)',
                    'Somatosensory and Motor Cortex', 
                    'Superior Parietal Cortex',
                    'Temporo-Parieto-Occipital Junction', 
                    'Ventral Stream Visual Cortex'
                    ]
        
        new_names = [
                    '23_inside', 
                    '19_cingulate_anterior_prefrontal_medial',
                    '11_auditory_association', 
                    '03_visual_dors',
                    '22_prefrontal_dorsolateral', 
                    '10_auditory_primary',
                    '02_visual_early',
                    '21_frontal_inferior',
                    '17_parietal_inferior', 
                    '12_insular_frontal_opercular', 
                    '14_lateral_temporal',
                    '05_visual_lateral',
                    '13_temporal_medial', 
                    '20_frontal_orbital_polar',
                    '07_paracentral_lob_mid_cingulate',
                    '18_cingulate_posterior',  
                    '09_opercular_posterior',
                    '08_premotor', 
                    '01_visual_primary', 
                    '06_somatosensory_motor', 
                    '16_parietal_superior', 
                    '15_temporal_parietal_occipital_junction', 
                    '04_visual_ventral', 
                    ]
        
        for o, n, i in zip(orig_names, new_names, ["%.2d" % i for i in range(23)]):
            os.rename(
                    os.path.join(subjects_dir, subj, 'label', '{}.HCPMMP1_combined-0{}.label'.format(hemi, i)), 
                    os.path.join(subjects_dir, subj, 'label', '{}.HCPMMP1_{}.label'.format(hemi, o)),
                    )
            os.rename(
                    os.path.join(subjects_dir, subj, 'label', '{}.HCPMMP1_{}.label'.format(hemi, o)), 
                    os.path.join(subjects_dir, subj, 'label', '{}.HCPMMP1_{}.label'.format(hemi, n)),
                    )