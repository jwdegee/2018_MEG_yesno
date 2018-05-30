#!/bin/bash
for subj in jw13 jw16 jw20 jw24
do
	path1=/home/jwdegee/degee/MEG/data/raw/mri/
	path2=.nii.gz
	path=$path1$subj$path2
	# recon-all -i $path -s $subj -all &
	recon-all -s $subj -all &
done