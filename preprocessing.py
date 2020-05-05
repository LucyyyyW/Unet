# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:38:28 2020

@author: DELL
"""

import SimpleITK as sitk
import sys,os
import numpy as np
import matplotlib.pyplot as plt
IMAGE_PATH = ''
IMAGE_FORMAT = '.png'
LABEL_PATH = ''
LABEL_FORMAT = '.png'

path_ = ''
path_lb = ''
#200=pool of left ventricle,500=myocardium of left ventricle,600=pool of right ventricle
LABEL_NUM = [200,500,600]
#img = sitk.ReadImage(path)
#data = sitk.GetArrayFromImage(img)
#all_one_array = np.ones_like(data[0])
def load_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data
'''
def clip_img(img,path):
    for i in range(img.shape[0])
        clip = img[i]
        clip = clip[0:240,0:240]
        clip_file = os.path.join(IMAGE_PATH,path+str(i)+IMAGE_FORMAT)
        plt.imsave(clip_file,clip)
'''
def clip_all(img,label,path1,path2):
    all_one_array = np.ones_like(img[0])
    a = ''
    for i in range(label.shape[0]):
        true_sum = np.sum(label[i] == LABEL_NUM[0]*all_one_array)+np.sum(label[i]==LABEL_NUM[1]*all_one_array)+np.sum(label[i]==LABEL_NUM[2]*all_one_array)
        if true_sum>0:
            clip = img[i]
            clip_ = label[i]
            clip = clip[100:340,100:340]
            clip_ = clip_[100:340,100:340]
            clip_file = os.path.join(IMAGE_PATH,path1+str(i)+ IMAGE_FORMAT)
            clip_file_ = os.path.join(LABEL_PATH,path2+str(i)+LABEL_FORMAT)
            plt.imsave(clip_file,clip)
            plt.imsave(clip_file_,clip_)
        else:
            a += str(i)
    return a
            
        
            
def main():
    path_ = 'D:/grade4.2/pj1/MSCMRSeg/MSCMRSeg/LGE/img'
    path_lb = 'D:/grade4.2/pj1/MSCMRSeg/MSCMRSeg/LGE/lab'
    dirs_ = os.listdir(path_)
    #dirs_ = dirs[0::3]
    dirs_lb = os.listdir(path_lb)
    total_list = []
    for i in range(len(dirs_)):
        img = load_img(os.path.join(path_,dirs_[i]))
        label = load_img(os.path.join(path_lb,dirs_lb[i]))
        a = clip_all(img,label,os.path.splitext(os.path.splitext(dirs_[i])[0])[0],os.path.splitext(os.path.splitext(dirs_lb[i])[0])[0])
        total_list.append(a)
    total_list = np.array(total_list)
    np.save('D:/grade4.2/pj1/list.npy',total_list)
    '''
    for dir in dirs_:
        img = load_img(os.path.join(path,dir))
        clip_img(img,os.path.splitext(os.path.splitext(dir)[0])[0])
    dirs_lb = os.listdir(path_lb)
    
    for dir in dirs_lb:
        img = load_img(os.path.join(path_lb,dir))
        clip_label(img,os.path.splitext(os.path.splitext(dir)[0])[0])
    '''
main()
    