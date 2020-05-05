# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:45:45 2020

@author: DELL
"""

import cv2
import numpy as np
import glob
PATH = 'D:/img'
SIZE = 240
def change(path):
    #path = 'D:/grade4.2/pj1/label1/aug_label_0_8741.png'
    img = cv2.imread(path)
    img1 = np.array(img)
    #img2= np.array(img)
    #print(img1[:,200,:])
    #print(img1[:,120,:])
    img1 = img1/255.
    #print(img)
    return img1
    #print(img)
    #img /= np.std(img,axis=0)
    #sum_ = np.sum(img==[84,1,68]*a)
    #print(sum_)
    #img
    #a = img[:,:,2]#.reshape(240,240,1)
    #a = a.reshape(240,240,1)
    #img = np.concatenate((img,a),axis=2)
    
    #print(img1[:,120])
    z = img1[:,:,0].reshape(240,240,1)#mask
    a = img2[:,:,0].reshape(240,240,1)#huang
    b = img1[:,:,2].reshape(240,240,1)#lan
    c = img2[:,:,2].reshape(240,240,1)
    
    #print(z[:,120])
    #a[a<50]=0
    z[z<80]=0
    z[z>90]=0
    z[z>0]=1
    #print(z[:,120])
    a[a<130]=0
    a[a>=150]=0
    a[a>0]=1
    b[b<=230]=0
    b[b>230]=1
    #print(b[:,120])
    #c[c<120]=0
    c[c>155]=0
    c[c<144]=0
    c[c>0]=1
    #print(c[:,200])
    img3 = np.concatenate((z,a),axis=2)
    img3 = np.concatenate((img3,b),axis=2)
    img3 = np.concatenate((img3,c),axis=2)
    #print(img3[:,120,:])
    #cv2.imwrite('D:/grade4.2/pj1/merge/2.png',img1)
    
    return img3
    
def main():
    paths = glob.glob(PATH+'/*.png')[:111]
    #paths = paths[800:]
    #print(paths[1])
    imgs = np.ndarray((len(paths),SIZE,SIZE,3),dtype=np.float32)
    i = 0
    for name in paths:
        img = change(name)
        imgs[i] = img
        #print(imgs[i])
        i+=1
    print(i)
    #print(imgs[5,:,160,:])
    np.save(PATH+'/T2try_img.npy',imgs)
main()