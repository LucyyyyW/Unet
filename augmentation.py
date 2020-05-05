# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:44:26 2020

@author: DELL
"""

import glob
import random
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
path = 'D:/img'
path_ = 'D:/label'
def load_data(path):
    img = load_img(path)
    img = img_to_array(img)
    img = img.reshape((1,)+img.shape)
    return img
def image_augmentation(img,label,augnum):#num means batch_size
    image_datagen = ImageDataGenerator(rotation_range = 0.2,
                                width_shift_range = 0.1,
                                 height_shift_range = 0.1,
                                 shear_range = 0.1,
                                 zoom_range = 0.1,
                                 fill_mode = 'nearest')
    label_datagen = ImageDataGenerator(rotation_range = 0.2,
                                width_shift_range = 0.1,
                                 height_shift_range = 0.1,
                                 shear_range = 0.1,
                                 zoom_range = 0.1,
                                 fill_mode = 'nearest')
    
    #random.seed(1)
    seed = random.randint(1,100000)
    n = 0
    for batch in image_datagen.flow(img,batch_size=1,save_to_dir=path,save_prefix='aug',save_format='png',seed=seed):
        n +=1
        if n > augnum:
            break
    n = 0
    
    for batch in image_datagen.flow(label,batch_size=1,save_to_dir=path_,save_prefix='aug_label',save_format='png',seed=seed):
        n +=1
        if n >augnum:
            break
    
    return

img_path = glob.glob(path+'/*.png')
#img_path = img_path[:5]
label_path = glob.glob(path_+'/*.png')
#label_path = label_path[:5]
for i in range(len(img_path)):
    img = load_data(img_path[i])
    label = load_data(label_path[i])
    image_augmentation(img,label,15)
'''
img = load_data(path)
image_augmentation(img,5)
'''