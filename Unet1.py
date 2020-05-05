# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:23:22 2020

@author: DELL
"""
import os
import glob
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import BatchNormalization,concatenate,Input, Conv2D, Activation,MaxPooling2D, Dense,UpSampling2D, Dropout, Cropping2D,Flatten,Add,Multiply
import random
from tensorflow.python import keras
from tensorflow.python.keras import backend 
import matplotlib.pyplot as plt
from tensorflow.python.keras import *
import math
SIZE = 240
IMG_PATH = './LGE'
LABEL_PATH = './LGE'
IMG_PATH1 = './data3'
LABEL_PATH1 = './data3'
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def scheduler(epoch):
    #lr = backend.get_value(model.optimizer.lr)
    # initial learningrate=0.01
    if epoch == 0:
        lr = 0.001
        return lr
    else:
        lr = backend.get_value(model.optimizer.lr)
        #backend.set_value(model.optimizer.lr,lr*math.exp(-0.3*epoch))
        backend.set_value(model.optimizer.lr,lr*math.exp(-0.3*epoch))
        print("lr changed to {}".format(lr*math.exp(-0.5*epoch)))
        return backend.get_value(model.optimizer.lr)
def load_data(path):
    img = load_img(path)
    img = img_to_array(img)
    img = img.reshape((1,)+img.shape)
    return img
def image_augmentation(img,label,augnum):#num means batch_size
    #label = label.reshape((1,)+label.shape)
    
    image_datagen = ImageDataGenerator(rotation_range = 0.2,
                                width_shift_range = 0.2,
                                 height_shift_range = 0.2,
                                 shear_range = 0.2,
                                 zoom_range = 0.2,
                                 fill_mode = 'nearest')
    label_datagen = ImageDataGenerator(rotation_range = 0.2,
                                width_shift_range = 0.2,
                                 height_shift_range = 0.2,
                                 shear_range = 0.2,
                                 zoom_range = 0.2,
                                 fill_mode = 'nearest')
    seed = random.randint(1,10000)
    n = 0
    seed_ = 1
    image_datagen.fit(img,seed=seed_)
    label_datagen.fit(label,seed=seed_)
    for batch in image_datagen.flow(img,batch_size=1,save_to_dir=IMG_PATH,save_prefix='aug',save_format='png',seed=seed):
        n +=1
        if n > augnum:
            break
    n = 0
    for batch in label_datagen.flow(label,batch_size=1,save_to_dir=LABEL_PATH,save_prefix='aug_label',save_format='png',seed=seed):
        n +=1
        if n >augnum:
            break
    return 
    '''
    seed = 1
    image_datagen.fit(img,augment=True,seed=seed)
    label_datagen.fit(label,augment=True,seed=seed)
    image_generator = image_datagen.flow_from_directory(IMG_PATH,class_mode=None,seed=seed)
    label_generator = label_datagen.flow_from_directory(LABEL_PATH,class_mode=None,seed=seed)
    train_generator = zip(image_generator,label_generator)
    return train_generator
    '''
def create_train_data():
    i = 0 
    print("Creating training images...")
    imgs = glob.glob(IMG_PATH +'/*.png')
    #labels = glob.glob(LABEL_PATH+'/*.png')
    imgdatas = np.ndarray((len(imgs),SIZE,SIZE,3),dtype=np.int8)
    #imglabels = np.ndarray((len(labels),SIZE,SIZE,4),dtype=np.int8)
    for imgname in imgs:
        img = load_img(imgname)
        img = img_to_array(img)
        img = img/255.
        img -= np.mean(img)
        img /= np.std(img,axis=0)
        imgdatas[i] = img
        i += 1
    i = 0
    '''
    for imgname in labels:
        img = load_img(imgname)
        img = img_to_array(img)
        img0 = img[:,:,2].reshape(240,240,1)#mask
        img1 = img[:,:,2].reshape(240,240,1)#yellow
        img2 = img[:,:,0].reshape(240,240,1)#blue
        img3 = img[:,:,1].reshape(240,240,1)#green
        img0[img0<80]=0
        img0[img0>90]=0
        img0[img0>0]=1
        img1[img1<130]=0
        img1[img1>150]=0
        img1[img1>0]=1
        img2[img2>=230]=1
        img2[img2<230]=0
        img3[img3>=200]=1
        img3[img3<200]=0
        img = np.concatenate((img0,img1),axis=2)
        img = np.concatenate((img,img2),axis=2)
        img = np.concatenate((img,img3),axis=2)
        #img /= np.std(img,axis=0)
        imglabels[i] = img
        i += 1
    '''
    print('loading done')
    np.save(IMG_PATH + '/imgs_train.npy',imgdatas)
    #np.save(LABEL_PATH + '/labels_train.npy',imglabels)
    print('saved')
    return
def load_train_data():
    print('loading')
    img_train = np.load(IMG_PATH1+'/imgs_train.npy')
    img_train -= np.mean(img_train2)
    img_train /= np.std(img_train2,axis=0)
    img_label = np.load(LABEL_PATH1+'/labels_train.npy')
    return img_train,img_label
def dice_coef(y_true,y_pred):
    sum1 = 2*tf.reduce_sum(y_true*y_pred,axis=(0,1,2))
    sum2 = tf.reduce_sum(y_true**2+y_pred**2,axis=(0,1,2))
    dice = (sum1+0.1)/(sum2+0.1)
    dice = tf.reduce_mean(dice)
    return dice
def dice_coef_loss(y_true,y_pred):
    return 1.-dice_coef(y_true,y_pred)
def dice_Score_0(y_true,y_pred):
    sum1 = tf.reduce_sum(y_true*y_pred,axis=(0,1,2))
    sum2 = tf.reduce_sum(y_true**2+y_pred**2,axis=(0,1,2))
    #sum2 = tf.reduce_sum(y_true**2,axis=(0,1,2))
    dice = 2*sum1/sum2
    return dice[0]
def dice_Score_1(y_true,y_pred):
    sum1 = tf.reduce_sum(y_true*y_pred,axis=(0,1,2))
    sum2 = tf.reduce_sum(y_true**2+y_pred**2,axis=(0,1,2))
    #sum2 = tf.reduce_sum(y_true**2,axis=(0,1,2))
    dice = 2*sum1/sum2
    return dice[1]
def dice_Score_2(y_true,y_pred):
    sum1 = tf.reduce_sum(y_true*y_pred,axis=(0,1,2))
    sum2 = tf.reduce_sum(y_true**2+y_pred**2,axis=(0,1,2))
    #sum2 = tf.reduce_sum(y_true**2,axis=(0,1,2))
    dice = 2*sum1/sum2
    return dice[2]
def dice_Score_3(y_true,y_pred):
    sum1 = tf.reduce_sum(y_true*y_pred,axis=(0,1,2))
    sum2 = tf.reduce_sum(y_true**2+y_pred**2,axis=(0,1,2))
    #sum2 = tf.reduce_sum(y_true**2,axis=(0,1,2))
    dice = 2*sum1/sum2
    return dice[3]
def Unet(num_class, image_size):

    inputs = Input(shape=[image_size, image_size, 3])
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(inputs)
    #conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    conv1= Conv2D(64,3,padding='same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv1)
    nor1 = BatchNormalization(momentum=.99,epsilon=0.001,
                               center=True,scale=True,beta_initializer='zeros',gamma_initializer='ones',
                               moving_mean_initializer='zeros',moving_variance_initializer='ones')(conv1)
    act1 = Activation('relu')(nor1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(pool1)
    #conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    conv2 = Conv2D(128,3,padding='same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv2)
    nor2 = BatchNormalization(momentum=.99,epsilon=0.001,
                               center=True,scale=True,beta_initializer='zeros',gamma_initializer='ones',
                               moving_mean_initializer='zeros',moving_variance_initializer='ones')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(nor2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(pool2)
    #conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    conv3 = Conv2D(256,3,padding='same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv3)
    nor3 = BatchNormalization(momentum=.99,epsilon=0.001,
                               center=True,scale=True,beta_initializer='zeros',gamma_initializer='ones',
                               moving_mean_initializer='zeros',moving_variance_initializer='ones')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(nor3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    #conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    conv4 = Conv2D(512,3,padding='same')(conv4)
    nor4 = BatchNormalization(momentum=.99,epsilon=0.001,
                               center=True,scale=True,beta_initializer='zeros',gamma_initializer='ones',
                               moving_mean_initializer='zeros',moving_variance_initializer='ones')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(nor4)
 
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(pool4)
    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    flatten1 = Flatten()(conv5)
    dense1 = Dense(1024,activation='relu')(flatten1)
    dense2 = Dense(256,activation='relu')(dense1)
    dense3 = Dense(3,activation='softmax')(dense2)
    conv5 = Conv2D(1024,3,padding='same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv5)
    nor5 = BatchNormalization(momentum=.99,epsilon=0.001,
                               center=True,scale=True,beta_initializer='zeros',gamma_initializer='ones',
                               moving_mean_initializer='zeros',moving_variance_initializer='ones')(conv5)
    #drop5 = Dropout(0.5)(nor5)
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(UpSampling2D(size = (2,2))(nor5))
    merge6 = concatenate([nor4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([nor3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([nor2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([nor1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer='glorot_normal',bias_initializer='zeros')(conv9)
#     conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(num_class, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = 'adam',
                 loss=dice_coef_loss, metrics = [dice_Score_0,dice_Score_1,dice_Score_2,dice_Score_3] )
    '''
    model = Model(inputs = [inputs], outputs =[conv10,dense3])
    model.compile(optimizer = 'adam',
                  loss={
                  'conv2d_23':dice_coef_loss,
                  'dense_2':'sparse_categorical_crossentropy'},
                  loss_weights={
                  'conv2d_23':0.5,
                  'dense_2':0.5},
                   metrics = {
                   'conv2d':[dice_Score_0,dice_Score_1,dice_Score_2,dice_Score_3],
                   'dense_2':['accuracy']})
    '''
    return model

def train():
    '''
    if os.path.exists(IMG_PATH+'/imgs_train.npy')==False:
        img_path = glob.glob(IMG_PATH+'/*.png')
        label_path = glob.glob(LABEL_PATH+'/*.png')
        print('loading')
        for i in range(len(img_path)):
            if i%10==0:
                print(i)
            img = load_data(img_path[i])
            label = load_data(label_path[i])
            image_augmentation(img,label,10)
    '''
   # create_train_data()
    img_train,img_label = load_train_data()
    print('loaded')
    global model
    model = Unet(4,240)
    #os.mkdir('my_log_dir_0330_1')
    model_checkpoint =  ModelCheckpoint('./SNn.h5', 
                                        monitor='loss',verbose=1, save_best_only=True)
    history = LossHistory()
    reduce_lr = LearningRateScheduler(scheduler)
    #callbacks = [
    #   keras.callbacks.TensorBoard(
    #            log_dir='my_log_dir',
    #            histogram_freq=1,
    #    ),history,reduce_lr,model_checkpoint]
    model.fit(img_train,img_label,batch_size=3,epochs=10,validation_split=0.1,
                        shuffle=True,verbose=1,callbacks=[reduce_lr,model_checkpoint])
    history.loss_plot('epoch')
    model.save('./SNn.h5')
    return

def save_img():
    imgs = np.load(Test_PATH + '/test_result.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save(Test_PATH+'/result%d.png'%(i))
    return 

train()
