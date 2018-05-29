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

# list copied from ls output:

heads = '''GU003_MRI_5_Electrodes.sip*
GU010_Big_Electrodes.sip*
GU011_T1_T2_AZ_CT3.sip*
GU012_Big_Electrodes.sip*
GU013_testing.sip*
GU015_AZ_CT_Electrodes.sip*
GU016_CT2_Electrodes.sip*
GU017_holesgray6_CT1_Electrodes.sip*
GU018_T2.sip*
GU019_Big_CT3_Electrodes.sip*
GU020_T1_CT2.sip*
GU021_Final_CT1_electrodes.sip*
GU024_CT2_electrodes.sip*
GU027__AA6_electrodes.sip*
NC004_Electrodes.sip*
NC010_Electrodes.sip*
NC011_PostPatching_CT3.sip*
NC012_full_CT4.sip*
NC013_CT1i.sip*
NC014_CT4.sip*
NC015_T1_T2_PatchedCT2.sip*
NC016_CT2_Electrodes.sip*
NC018_AZ_8_25_17_CT1_test_delete.sip*
NC019_AA10_electrodes.sip*
NC01_CT2.sip*
NC020_done_CT_electrodes.sip*
NC021_CT1_electrodes.sip*
NC022_CT2_electrodes.sip*
NC023__AA10_electrodes.sip*]'''

heads = heads.split('\n')

for head in heads:
    heads[heads.index(head)] = head.split('_')[0]
    
'''
os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads')
'''
  
def create_text_file_per_channel(maps):
    maps = open(maps)
    maps = maps.readlines()

    

    
    myMaps = []
    for i in range(len(maps)):
        myMaps.append( maps[i][:-1].split('.')[0] + '_of.nii')
        
    gm = [x for x in myMaps if (any(substring in x.lower() for substring in ['gray','_gm'])) ]
    
    wm = [x for x in myMaps if (any(substring in x.lower() for substring in ['white', '_wm'])) ]
    cs = [x for x in myMaps if (any(substring in x.lower() for substring in ['csf'])) ]
    ai = [x for x in myMaps if (any(substring in x.lower() for substring in ['air'])) ]
    sk = [x for x in myMaps if (any(substring in x.lower() for substring in ['skin'])) ]
    bo = [x for x in myMaps if (any(substring in x.lower() for substring in ['bone', 'skull'])) ]

    

    return gm, wm, cs, ai, sk, bo


'''
os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads')
print 'Creating text file containing all channels ...'
gray = []
white = []
csf = []
air = []
skin = []
bone = []

for head in heads:
    os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads')
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
        f = open('/home/lukas/Documents/projects/strokeHeads/labels_{}.txt'.format(channel), 'a')
        for item in channels[channel]:
            f.write("%s\n" % item)
        f.close()
        os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads')
        
        
############# Get MRIs ###################################################################
        
MRIs = []
for head in heads:
    os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads')
    maps = os.getcwd() + '/' + head + '/' + 'masks.txt'
    address = '/'.join(maps.split('/')[:-1])
    os.chdir(address)
    
    MRI =  [x for x in os.listdir(address) if (any(substring in x.lower() for substring in ['background1'])) & (not (any(substring in x.lower() for substring  in ['log'])))]
    for i in range(len(MRI)):
        MRI[i] = os.getcwd() + '/' + MRI[i]
    MRIs.extend(MRI)
    
len(MRIs)
MRIs



f = open('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/T1.txt', 'a')
for item in MRIs:
    f.write("%s\n" % item)
f.close()
os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads')


#%%##########################   Add prefix r (done after resampling in MATLAB)

channels = ['labels_Air','labels_Bone','labels_CSF','labels_GM','labels_Skin','labels_WM']

for j in range(len(channels)):
    resampled = '/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/{}.txt'.format(channels[j])
    img = open(resampled)
    img = img.readlines()
    
    for i in range(len(img)):
        img[i] = '/'.join(img[i].split('/')[0:-1]) + '/r' + img[i].split('/')[-1]
    
    img
    f = open('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/resampled_{}.txt'.format(channels[j]), 'a')
    for item in img:
        f.write("%s" % item)
    f.close()
    
#%%##                       Create new text file per subject folder, containing address for resampled masks
    
for head in heads:
    os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads')
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
    os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads')
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

os.chdir('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/')
files = ['labels_GM','labels_WM','labels_CSF','labels_Bone','labels_Skin','labels_Air']
channels = [0]*6
for j in range(len(files)):
    channels[j] = '/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/resampled_{}.txt'.format(files[j])       

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
    channels[j] = '/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/resampled_of_{}.txt'.format(files[j])

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
    
fi = open('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/all_labels.txt')
lines = fi.readlines()
fi.close()
for i in range(len(lines)):
    lines[i] = '/'.join(lines[i].split('/')[0:-1]) + '/LABELS.nii'
    
fi = open('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/all_labels.txt', 'a')
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

fi = open('/home/lukas/Documents/projects/minimallyConciousMRIstDCS/MRI_LucasParra/niftis/T1.txt')
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
    
fi = open('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/resampled_T1.txt')
lines = fi.readlines()
fi.close()
for i in range(len(lines)):
    lines[i] = '/'.join(lines[i].split('/')[0:-1]) + '/n' + lines[i].split('/')[-1]
    
fi = open('/home/lukas/Documents/projects/strokeHeads/SegmentedStrokeHeads/Data_DeepMedic/nr_T1', 'a')
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

nii = '/media/lukas/d37fc604-163d-4e04-83de-88993c28e419/home/lukas/Documents/StrokeHeads/DATA_coreg/GU003_GU003_Background1_Background 3_COPY.nii'
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
MRIs_training = '/media/lukas/d37fc604-163d-4e04-83de-88993c28e419/home/lukas/Documents/StrokeHeads/DATA_coreg/data_after_coreg.txt'
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
test_set = open('/home/lukas/Documents/projects/minimallyConciousMRIstDCS/MRI_LucasParra/niftis/SEGMENTATION_CNN/coreg_T1.txt')
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