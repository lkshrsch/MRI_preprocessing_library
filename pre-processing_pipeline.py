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


os.chdir('/home/hirsch/Documents/projects/brainSegmentation/')

# UNZIP HEADS

# CREATE TEXT FILES WITH MRIS TO PREPROCESS

from MRI_preprocessing_lib import generate_MRIs_textfile

wd = '/home/hirsch/Documents/projects/strokeHeads/raw/Batch2/'

generate_MRIs_textfile(wd)


#%% COREGISTRATION

"MATLAB SCRIPT"

#%% RESCALING

print('Starting image rescaling.')

os.chdir('/home/hirsch/Documents/projects/brainSegmentation/')


wd = '/home/hirsch/Documents/projects/strokeHeads/preprocessing_pipeline/'


from MRI_preprocessing_lib import rescale

TEXT_FILE_MRIs = '/home/hirsch/Documents/projects/strokeHeads/preprocessing_pipeline/data_after_coreg.txt'

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


#%% ZERO PADDING

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


#%% STANDARDIZATION

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


#%%###########################################################################################################

#---------------------------------   PREPROCESSING OF LABELS 

#%% GENERATE TEXT FILES FOR CHANNELS
import os
os.chdir('/home/hirsch/Documents/projects/brainSegmentation/')
from MRI_preprocessing_lib import create_masks_text_per_subject
wd = '/home/hirsch/Documents/projects/strokeHeads/raw/Batch2/'
create_masks_text_per_subject(wd)



#%% CHECK OVERLAPS
os.chdir('/home/hirsch/Documents/projects/brainSegmentation/')

from MRI_preprocessing_lib import add_MRI_maps_checkOverlap, get_heads
wd = '/home/hirsch/Documents/projects/strokeHeads/raw/Batch2/'
os.chdir(wd)    

heads = get_heads(wd)

print 'Counting how many voxels of each segmentation mask overlap in each head ...'
for head in heads:
    
    maps = os.getcwd() + '/' + head + '/' + 'masks.txt'
    address = '/'.join(maps.split('/')[:-1])
    os.chdir(address)
    overlap, count = add_MRI_maps_checkOverlap(maps)
    name = ''.join(maps.split('/')[-2])
    
    
    string = name + ' ' + str(overlap) + ' : ' + str(count) + ' Voxels.'  
    
    f = open(wd + '/overlaps_resampled_heads.txt', 'a')
    f.write('\n' + string)
    f.close()
    os.chdir(wd)


# RESOLVE OVERLAPS  --- MATLAB



#%% CHECK OVERLAPS
os.chdir('/home/hirsch/Documents/projects/brainSegmentation/')

from MRI_preprocessing_lib import add_MRI_maps_checkOverlap, get_heads
wd = '/home/hirsch/Documents/projects/strokeHeads/raw/Batch2/'
os.chdir(wd)    

heads = get_heads(wd)

print 'Counting how many voxels of each segmentation mask overlap in each head ...'
for head in heads:
    
    maps = os.getcwd() + '/' + head + '/' + 'masks.txt'
    address = '/'.join(maps.split('/')[:-1])
    os.chdir(address)
    overlap, count = add_MRI_maps_checkOverlap(maps, overlap_free=True)
    name = ''.join(maps.split('/')[-2])
    
    
    string = name + ' ' + str(overlap) + ' : ' + str(count) + ' Voxels.'  
    
    f = open(wd + '/overlaps_resampled_heads.txt', 'a')
    f.write('\n' + string)
    f.close()
    os.chdir(wd)


#%% GENERATE TARGET LABELS BY ADDING LABELS 
import os
import nibabel as nib
os.chdir('/home/hirsch/Documents/projects/brainSegmentation/')

from MRI_preprocessing_lib import  addMRIs_create_Target, get_heads

wd = '/home/hirsch/Documents/projects/strokeHeads/raw/Batch2/'

os.chdir(wd)    
print 'Creating Nifti image of overlaps in masks in each head ...'

heads = get_heads(wd)

for head in heads:
    maps = wd + '/{}/masks.txt'.format(head)
    address = '/'.join(maps.split('/')[:-1])
    os.chdir(address)
    img = addMRIs_create_Target(maps)
    out = os.getcwd() + '/LABEL.nii'
    nib.save(img, out)
    os.chdir(wd)    


#-------------- END------------
# COREG (TOGETHER WITH MRI)
# PADDING (TOGETHER WITH MRI)