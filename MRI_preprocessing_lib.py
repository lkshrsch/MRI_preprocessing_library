#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 18:06:35 2018

Sort all stroke lesion patients into text files for usage with deepmedic

@author: lukas
"""



import nibabel as nib
import numpy as np
import os


def create_masks_text_per_subject(wd):
    
    heads = get_heads(wd)    

    for i in range(len(heads)):
        print('Creating masks.txt for subject {}'.format(heads[i]))
        os.chdir(wd + heads[i])        

        files = os.listdir(os.getcwd())        
        
        gm = [x for x in files if (any(substring in x.lower() for substring in ['gray','_gm'])) ]    
        wm = [x for x in files if (any(substring in x.lower() for substring in ['white', '_wm'])) ]
        cs = [x for x in files if (any(substring in x.lower() for substring in ['csf'])) ]
        ai = [x for x in files if (any(substring in x.lower() for substring in ['air'])) ]
        sk = [x for x in files if (any(substring in x.lower() for substring in ['skin'])) ]
        bo = [x for x in files if (any(substring in x.lower() for substring in ['bone', 'skull'])) ]

        masks = [ai, gm, wm, cs, bo, sk]
        masks.sort() # For input into MATLAB solve overlaps

        f = open(os.getcwd() + '/masks.txt', 'w')
        for item in masks:
            mask = os.getcwd() + '/' + item[0]
            f.write("%s\n" % mask)
        f.close()

    

def get_heads(wd):
    os.chdir(wd)
    extracted_heads = [d for d in os.listdir(wd) if len(d.split('.')) == 1]
    extracted_heads.sort()
    return extracted_heads

def generate_MRIs_textfile(wd):
    os.chdir(wd)
    extracted_heads = get_heads(wd)
    mris = []
    for head in extracted_heads:
        os.chdir(wd+head)
        files = os.listdir(os.getcwd())
        mri = [d for d in files if (any(substring in d.lower() for substring in ['background'])) & (not (any(substring in d.lower() for substring  in ['blank', 'log'])))]
        mri = mri[0]
        mris.append(os.getcwd() +'/'+ mri)
    
    f = open(wd + '/MRIs.txt', 'a')
    for item in mris:
        f.write("%s\n" % item)
    f.close()
    print('Text file created with MRI addresses on {}'.format(wd))
    
def update_sum_N(nii, m, N):
    img = nib.load(nii)
    data = img.get_data()
    m = m + np.sum(data)
    N = N + reduce(lambda x,y: x*y, data.shape)
    return m, N

def update_square_diffs(nii, mean, diffs):
    img = nib.load(nii)
    data = img.get_data()
    difs = np.array((data-mean)**2, dtype='uint64')
    diffs = diffs + np.sum(difs)
    return diffs
    

def get_overall_mean_std_training_set(MRIs):
    m = 0
    N = 0
    fi = open(MRIs)
    lines = fi.readlines()
    for nii in lines:
        m, N = update_sum_N(nii[:-1], m, N)
    mean = float(m)/N
    diffs = np.array(0, dtype='uint64')
    for nii in lines:
        diffs = update_square_diffs(nii[:-1], mean, diffs)
    s = np.sqrt(diffs/N)
    return mean, s

def normalizeMRI(nii, mean, std):
    img = nib.load(nii)
    data = img.get_data()
    aff = img.affine
    data1 = (data - mean)/std
    out = nib.Nifti1Image(data1, aff)
    return(out)
    
def standardize_MRIs(MRIs, OUT_FILE, mean=None, std=None):
    if (mean == None) & (std == None):
        mean, std = get_overall_mean_std_training_set(MRIs)
    mris = open(MRIs)
    lines = mris.readlines()
    mris.close()
    fi = open(OUT_FILE + 'stand_padded_rescaled_coreg_MRIs.txt', 'a')
    for nii in lines:
        nii = nii[0:-1]
        img = normalizeMRI(nii, mean, std)
        img_out = OUT_FILE + nii.split('.')[0].split('/')[-1] + '_stand.nii'
        nib.save(img, img_out)   
        fi.write("{}\n".format(img_out))
    fi.close()
    fi = open(OUT_FILE + 'MEAN_STD.txt', 'a')
    fi.write("{}".format([mean, std]))
    fi.close()




#%%########             Create zero border for deep medic


def padd_image(d, size = 52, tpm_flag = False):
    "For usage with patch-based architecture. Window size of 52 based on the deepMedic border size"
        
    if len(d.shape) < 4: 
        chn = 1
        d = d.reshape((d.shape[0] ,d.shape[1] ,d.shape[2] , (1)))
    else:
        chn = d.shape[3]
    dims = (d.shape[0] + size, d.shape[1] + size, d.shape[2] + size,chn )
    
    padded = np.zeros(dims) + np.mean(d[0,0])
    if tpm_flag:
        padded[:,:,:,0] = padded[:,:,:,0]*0 + 0.99
    for ch in range(chn):
        padded[(size//2):(size//2 + d.shape[0]),(size//2):(size//2 + d.shape[1]), (size//2):(size//2 + d.shape[2]), ch] = d[:,:,:,ch]
    
    if chn == 1:
        padded = padded.reshape(padded.shape[0],padded.shape[1],padded.shape[2])
    
    return padded



def rescale(img, min_new = 0, max_new = 255):
    " Rescale image. Default is int8 [0-255] "
    return ((img - img.min()) * (float(max_new - min_new) / float(img.max() - img.min()))) + min_new
    
    
    
def add_MRI_maps_checkOverlap(maps, overlap_free = False):
    maps = open(maps)
    maps = maps.readlines()
    niftiArray = []
    for i in range(len(maps)):
        if overlap_free:
            myMap = maps[i][:-1].split('.')[0] + '_of.nii' 
        else:
            myMap = maps[i][:-1]
   # For overlap free masks
        #myMap = '/'.join(myMap.split('/')[0:-1]) +'/r' +  myMap.split('/')[-1]   # for resampled masks
        niftiArray.append(nib.load(myMap))
    img = 0
    count = 0
    overlap = False
    #overlapping_tissues = []
    for i in range(len(niftiArray)):
        new_mask = niftiArray[i].get_data()
        
        #new_mask[new_mask < np.max(new_mask)] = 0
        new_mask[new_mask != 0] = 1
        
        new_mask = np.array(np.array(new_mask ,dtype='?'),dtype='int16')
        #new_mask =  np.array(np.array(niftiArray[i].get_data()),'int16')
        
        img = img + new_mask

    count = np.sum(img > 1)
    if len(np.unique(img)) > 2:#len(niftiArray):
         overlap = True
         
    return overlap, count#, overlapping_tissues



def addMRIs_create_Target(maps):
    maps = open(maps)
    myMaps = maps.readlines()
    myMaps = [x.split('.')[0:-1][0] + '_of.nii' for x in myMaps]
        
    
    
    gm = [x for x in myMaps if (any(substring in x.lower() for substring in ['gray','_gm'])) ]
    wm = [x for x in myMaps if (any(substring in x.lower() for substring in ['white', '_wm'])) ]
    cs = [x for x in myMaps if (any(substring in x.lower() for substring in ['csf'])) ]
    ai = [x for x in myMaps if (any(substring in x.lower() for substring in ['air'])) ]
    sk = [x for x in myMaps if (any(substring in x.lower() for substring in ['skin'])) ]
    bo = [x for x in myMaps if (any(substring in x.lower() for substring in ['bone', 'skull'])) ]
    
    niftiArray = []
    niftiArray.append(nib.load(ai[0]))
    niftiArray.append(nib.load(gm[0]))
    niftiArray.append(nib.load(wm[0]))
    niftiArray.append(nib.load(cs[0]))
    niftiArray.append(nib.load(bo[0]))
    niftiArray.append(nib.load(sk[0]))
    
    img = 0
    
    for i in range(len(niftiArray)):        
        tissue_index = i
        
        #new_mask = niftiArray[i].get_data()
        #new_mask[new_mask < np.max(new_mask)] = 0
        #new_mask = np.array(np.array(new_mask ,dtype='?'),dtype='int16')*tissue_index
        new_mask =  np.array(np.array(niftiArray[i].get_data()),'int16')        
        img = img + new_mask*tissue_index   

    final = nib.Nifti1Image(img, niftiArray[0].affine)
    return final

