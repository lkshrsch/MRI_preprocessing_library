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

    

'''
os.chdir('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads')
print 'Creating text file containing all channels ...'
gray = []
white = []
csf = []
air = []
skin = []
bone = []

for head in heads:
    os.chdir('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads')
    maps = os.getcwd() + '/' + head + '/' + 'masks.txt'
    address = '/'.join(maps.split('/')[:-1])
    os.chdir(address)
    
    gm, wm, cs, ai, sk, bo = create_text_file_per_channel(maps)
    
    gray.extend(gm)
    white.extend(wm)
    csf.extend(cs)
    air.extend(ai)
    skin.extend(sk)
    bone.extend(bo)
    
    channels = {'GM':gray, 'WM':white, 'CSF':csf, 'Air':air, 'Skin':skin, 'Bone':bone}
    
    for channel in channels.keys():
        print channel
        f = open('/home/hirsch/Documents/projects/strokeHeads/labels_{}.txt'.format(channel), 'a')
        for item in channels[channel]:
            f.write("%s\n" % item)
        f.close()
        os.chdir('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads')
        
'''

#wd = '/home/hirsch/Documents/projects/strokeHeads/raw/Batch2/'

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
    
'''
#%%##########################   Add prefix r (done after resampling in MATLAB)

channels = ['labels_Air','labels_Bone','labels_CSF','labels_GM','labels_Skin','labels_WM']

for j in range(len(channels)):
    resampled = '/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/{}.txt'.format(channels[j])
    img = open(resampled)
    img = img.readlines()
    
    for i in range(len(img)):
        img[i] = '/'.join(img[i].split('/')[0:-1]) + '/r' + img[i].split('/')[-1]
    
    img
    f = open('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/resampled_{}.txt'.format(channels[j]), 'a')
    for item in img:
        f.write("%s" % item)
    f.close()
    
#%%##                       Create new text file per subject folder, containing address for resampled masks
    
for head in heads:
    os.chdir('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads')
    maps = os.getcwd() + '/' + head + '/' + 'masks.txt'
    address = '/'.join(maps.split('/')[:-1])
    os.chdir(address)
    m = open(maps)
    masks = m.readlines()
    
    f = open('rMasks.txt', 'a')
    for mask in masks:
        rMask = '/'.join(mask.split('/')[0:-1]) + '/r' + mask.split('/')[-1].split('.')[0] + '_of.nii'
        f.write("%s\n" % rMask)
    f.close()
    
#%%                     Go to all resampled masks, and convert all nonzero values to 1

import shutil

for head in heads:
    os.chdir('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads')
    maps = os.getcwd() + '/' + head + '/' + 'rMasks.txt'
    address = '/'.join(maps.split('/')[:-1])
    os.chdir(address)
    m = open(maps)
    masks = m.readlines() 
    
    for mask in masks:
        mask = mask[0:-1]
        copy = mask.split('.')[0] + '_(bkp).nii' 
        shutil.copy2(mask, copy)
        nifti = nib.load(mask)
        img = nifti.get_data()
        img[img != 0] = 1
        out = nib.Nifti1Image(img, nifti.affine)
        os.remove(mask)
        nib.save(out, mask)


#%%#     Create text file with resampled overlap free channels

os.chdir('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/')
files = ['labels_GM','labels_WM','labels_CSF','labels_Bone','labels_Skin','labels_Air']
channels = [0]*6
for j in range(len(files)):
    channels[j] = '/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/resampled_{}.txt'.format(files[j])       

for j in range(6):
    maps = open(channels[j]).readlines()
    for i in range(len(maps)):
        maps[i] = maps[i].split('.')[0] + '_rof.nii'
        
    f = open('resampled_of_{}.txt'.format(files[j]), 'a')
    for mask in maps:
        f.write("%s\n" % mask)
    f.close()
        


#%%###   Add all label channels into single files


files = ['labels_Air','labels_GM','labels_WM','labels_CSF','labels_Bone','labels_Skin']  # Air needs to be class 0, because often air sorrounding head is NOT labeled, but retains value 0 because I initialize it that way
channels = [0]*6
for j in range(len(files)):
    channels[j] = '/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/resampled_of_{}.txt'.format(files[j])

for subj in range(29):
    img = 0
    for ch in range(6):
        current_ch = open(channels[ch]).readlines()[subj][0:-1]
        nii = nib.load(current_ch)
        img = img + (nii.get_data()*ch)
    assert np.sum(np.unique(img)) == 15, 'Overlap! {}  {}'.format(subj,ch)
    path = '/'.join(current_ch.split('/')[0:-1])
    if os.path.exists(path + '/LABELS.nii'):
        os.remove(path + '/LABELS.nii')
    out = nib.Nifti1Image(img, nii.affine)
    
    nib.save(out, path + '/LABELS.nii')
        
        
        
    # 0 = air , 1 = GM, 2 = WM, 3 = CSF, 4 = Bone, 5 = Skin, 6 = air
    
fi = open('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/all_labels.txt')
lines = fi.readlines()
fi.close()
for i in range(len(lines)):
    lines[i] = '/'.join(lines[i].split('/')[0:-1]) + '/LABELS.nii'
    
fi = open('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/all_labels.txt', 'a')
for item in lines:
    fi.write("%s\n" % item)
fi.close()
'''

#%%#########################  Normalize Intensities 
'''
def normalizeMRI(nii):
    img = nib.load(nii)
    data = img.get_data()
    aff = img.affine
    data1 = np.ma.masked_array(data, data==0)
    m = data1.mean()
    s = data1.std()
    data1 = (data1 - m)/s
    data1 = np.ma.getdata(data1)
    out = nib.Nifti1Image(data1, aff)
    return(out)

fi = open('/home/hirsch/Documents/projects/minimallyConciousMRIstDCS/MRI_LucasParra/niftis/T1.txt')
lines = fi.readlines()
fi.close()

for nii in lines:
    out_path = '/'.join(nii.split('/')[0:-1]) + '/n' + nii.split('/')[-1][0:-1]
    if os.path.exists(out_path):
        os.remove(out_path)
    nii = nii[0:-1]
    out = normalizeMRI(nii)
    nib.save(out, out_path)

# create file
    
fi = open('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/resampled_T1.txt')
lines = fi.readlines()
fi.close()
for i in range(len(lines)):
    lines[i] = '/'.join(lines[i].split('/')[0:-1]) + '/n' + lines[i].split('/')[-1]
    
fi = open('/home/hirsch/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/nr_T1', 'a')
for item in lines:
    fi.write("%s" % item)
fi.close()

def normalize_training_set_MRI(nii):
    img = nib.load(nii)
    data = img.get_data()
    aff = img.affine
    data1 = np.ma.masked_array(data, data==0)
    m = data1.mean()
    s = data1.std()
    data1 = (data1 - m)/s
    data1 = np.ma.getdata(data1)
    out = nib.Nifti1Image(data1, aff)
    return(out)

nii = '/media/hirsch/d37fc604-163d-4e04-83de-88993c28e419/home/hirsch/Documents/StrokeHeads/DATA_coreg/GU003_GU003_Background1_Background 3_COPY.nii'
'''
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
'''
MRIs_training = '/media/hirsch/d37fc604-163d-4e04-83de-88993c28e419/home/hirsch/Documents/StrokeHeads/DATA_coreg/data_after_coreg.txt'
MEAN, STD = get_overall_mean_std_training_set(MRIs_training)
'''
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

'''
test_set = open('/home/hirsch/Documents/projects/minimallyConciousMRIstDCS/MRI_LucasParra/niftis/SEGMENTATION_CNN/coreg_T1.txt')
lines = test_set.readlines()
test_set.close()

for nii in lines:
    out_path = '/'.join(nii.split('/')[0:-1]) + '/normalized/n' + nii.split('/')[-1][0:-1]
    if os.path.exists(out_path):
        os.remove(out_path)
    out_folder = '/'.join(nii.split('/')[0:-1]) + '/normalized/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    nii = nii[0:-1]
    out = normalizeMRI(nii, MEAN, STD)
    nib.save(out, out_path)
    
'''



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

