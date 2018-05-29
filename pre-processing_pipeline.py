#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:09:50 2018


Peprocessing pipeline for input MRIs to Multiscale_TPM CNN

In reference to a defined TPM reference

STEPS:
    
Coregistration (TPM) --> Rescaling (int8) --> Zero Padding --> Standardization (training set) 

@author: lukas
"""
import nibabel as nib
import numpy as np
import os


# COREGISTRATION

"MATLAB SCRIPT"

# RESCALING

print('Starting image rescaling.')

os.chdir('/home/lukas/Documents/projects/brainSegmentation/')


wd = '/home/lukas/Documents/projects/strokeHeads/preprocessing_pipeline/'


from MRI_preprocessing_lib import rescale

TEXT_FILE_MRIs = '/home/lukas/Documents/projects/strokeHeads/preprocessing_pipeline/data_after_coreg.txt'

RESCALED_ADDRESS = wd + 'rescaled_coreg_T1/'

if not os.path.exists(RESCALED_ADDRESS):
    os.mkdir(RESCALED_ADDRESS)
    fi = open(RESCALED_ADDRESS + 'rescaled_coreg_MRIs.txt', 'a')

f = open(TEXT_FILE_MRIs)
l = f.readlines()

for i in range(len(l)):
    nifti = l[i][:-1]
    nii = nib.load(nifti)
    img = nii.get_data()
    print('{}'.format([np.min(img), np.max(img)]))
    
    img = np.array(rescale(img),dtype='uint8')
    print('{}'.format([np.min(img), np.max(img)]))
    img = nib.Nifti1Image(img, nii.affine)
    img_out = RESCALED_ADDRESS + nifti.split('.')[0].split('/')[-1] + '_int8.nii'
    nib.save(img, img_out )
    fi.write("{}\n".format(img_out))
fi.close()
print('End of image rescaling.')


# ZERO PADDING

print('Starting image padding.')

from MRI_preprocessing_lib import padd_image

TEXT_FILE_MRIs = RESCALED_ADDRESS + 'rescaled_coreg_MRIs.txt'

PADDED_ADDRESS = wd + 'padded_rescaled_coreg_T1/'

if not os.path.exists(PADDED_ADDRESS):
    os.mkdir(PADDED_ADDRESS)
    fi = open(PADDED_ADDRESS + 'padded_rescaled_coreg_MRIs.txt', 'a')

f = open(TEXT_FILE_MRIs)
l = f.readlines()

for i in range(len(l)):
    nifti = l[i][:-1]
    nii = nib.load(nifti)
    d = nii.get_data()
    padded = padd_image(d, tpm_flag=False)
    img = nib.Nifti1Image(padded, nii.affine)
    img_out = PADDED_ADDRESS + nifti.split('.')[0].split('/')[-1] + '_padded.nii'
    nib.save(img, img_out )
    fi.write("{}\n".format(img_out))
fi.close()
print('End of image padding.')


# STANDARDIZATION

print('Starting standardization.')

from MRI_preprocessing_lib import standardize_MRIs

TEXT_FILE_MRIs = PADDED_ADDRESS + 'padded_rescaled_coreg_MRIs.txt'

STAND_ADDRESS = wd + 'stand_pad_rescaled_coreg_T1/'

if not os.path.exists(STAND_ADDRESS):
    os.mkdir(STAND_ADDRESS)
    
MEAN = 27.39592727283753
STD = 55.587982695829403
    
standardize_MRIs(TEXT_FILE_MRIs, STAND_ADDRESS, mean=MEAN, std=STD)



print('End standardization.')
