from __future__ import absolute_import, division, print_function, unicode_literals

#@markdown ##Load key 3D U-Net dependencies and instantiate network
Notebook_version = '2.2.1'
Network = 'U-Net (3D)'

from builtins import any as b_any

#Put the imported code and libraries here
# !pip install fpdf

import elasticdeform

import tifffile

import imgaug.augmenters as iaa

import os
import csv
import random
import h5py
import imageio
import math
import shutil
import argparse
import neptune


import pandas as pd
from glob import glob
from tqdm import tqdm

from skimage import transform
from skimage import exposure
from skimage import color
from skimage import io

from scipy.ndimage import zoom

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))

# from keras import backend as K

# from keras.layers import Conv3D
# from keras.layers import BatchNormalization
# from keras.layers import ReLU
# from keras.layers import MaxPooling3D
# from keras.layers import Conv3DTranspose
# from keras.layers import Input
# from keras.layers import Concatenate

# from keras.models import Model

# from keras.utils import Sequence
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import CSVLogger
# from keras.callbacks import Callback

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate

from tensorflow.keras.models import Model

from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from ipywidgets import interact
from ipywidgets import interactive
from ipywidgets import fixed
from ipywidgets import interact_manual
import ipywidgets as widgets

from fpdf import FPDF, HTMLMixin
from datetime import datetime
import subprocess
from pip._internal.operations.freeze import freeze
import time

from skimage import io
import matplotlib

from skimage import io
from shutil import rmtree


print("Dependencies installed and imported.")

# neptune setup

EXPERIMENT_DESCRIPTION = 'Track the training details'

neptune_run = neptune.init_run(
    project="BroadImagingPlatform/DBP-Doe",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMWU0MmFiZS0yZGVkLTQwMGItYTczNC0yNzdiNTljMTExY2QifQ==",
)

#Create a variable to get and store relative base path
base_path = os.getcwd()

# Define MultiPageTiffGenerator class
class MultiPageTiffGenerator(Sequence):

    def __init__(self,
                 source_path,
                 target_path,
                 batch_size=1,
                 shape=(128,128,32,1),
                 augment=False,
                 augmentations=[],
                 deform_augment=False,
                 deform_augmentation_params=(5,3,4),
                 val_split=0.2,
                 is_val=False,
                 random_crop=True,
                 downscale=1,
                 binary_target=False):

        # If directory with various multi-page tiffiles is provided read as list
        if os.path.isfile(source_path):
            self.dir_flag = False
            self.source = tifffile.imread(source_path)
            if binary_target:
                self.target = tifffile.imread(target_path).astype(bool)
            else:
                self.target = tifffile.imread(target_path)

        elif os.path.isdir(source_path):
            self.dir_flag = True
            self.source_dir_list = glob(os.path.join(source_path, '*'))
            self.target_dir_list = glob(os.path.join(target_path, '*'))

            self.source_dir_list.sort()
            self.target_dir_list.sort()

        self.shape = shape
        self.batch_size = batch_size
        self.augment = augment
        self.val_split = val_split
        self.is_val = is_val
        self.random_crop = random_crop
        self.downscale = downscale
        self.binary_target = binary_target
        self.deform_augment = deform_augment
        self.on_epoch_end()

        if self.augment:
            # pass list of augmentation functions
            self.seq = iaa.Sequential(augmentations, random_order=True) # apply augmenters in random order
        if self.deform_augment:
            self.deform_sigma, self.deform_points, self.deform_order = deform_augmentation_params

    def __len__(self):
        # If various multi-page tiff files provided sum all images within each
        if self.augment:
            augment_factor = 4
        else:
            augment_factor = 1

        if self.dir_flag:
            num_of_imgs = 0
            for tiff_path in self.source_dir_list:
                num_of_imgs += tifffile.imread(tiff_path).shape[0]
            xy_shape = tifffile.imread(self.source_dir_list[0]).shape[1:]

            if self.is_val:
                if self.random_crop:
                    crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                    volume = xy_shape[0] * xy_shape[1] * self.val_split * num_of_imgs
                    return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))
                else:
                    return math.floor(self.val_split * num_of_imgs / self.batch_size)
            else:
                if self.random_crop:
                    crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                    volume = xy_shape[0] * xy_shape[1] * (1 - self.val_split) * num_of_imgs
                    return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))

                else:
                    return math.floor(augment_factor*(1 - self.val_split) * num_of_imgs/self.batch_size)
        else:
            if self.is_val:
                if self.random_crop:
                    crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                    volume = self.source.shape[0] * self.source.shape[1] * self.val_split * self.source.shape[2]
                    return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))
                else:
                    return math.floor((self.val_split * self.source.shape[0] / self.batch_size))
            else:
                if self.random_crop:
                    crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                    volume = self.source.shape[0] * self.source.shape[1] * (1 - self.val_split) * self.source.shape[2]
                    return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))
                else:
                    return math.floor(augment_factor * (1 - self.val_split) * self.source.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        source_batch = np.empty((self.batch_size,
                                 self.shape[0],
                                 self.shape[1],
                                 self.shape[2],
                                 self.shape[3]))
        target_batch = np.empty((self.batch_size,
                                 self.shape[0],
                                 self.shape[1],
                                 self.shape[2],
                                 self.shape[3]))

        for batch in range(self.batch_size):
            # Modulo operator ensures IndexError is avoided
            stack_start = self.batch_list[(idx+batch*self.shape[2])%len(self.batch_list)]

            if self.dir_flag:
                self.source = tifffile.imread(self.source_dir_list[stack_start[0]])
                if self.binary_target:
                    self.target = tifffile.imread(self.target_dir_list[stack_start[0]]).astype(bool)
                else:
                    self.target = tifffile.imread(self.target_dir_list[stack_start[0]])

            src_list = []
            tgt_list = []
            for i in range(stack_start[1], stack_start[1]+self.shape[2]):
                src = self.source[i]
                src = transform.downscale_local_mean(src, (self.downscale, self.downscale))
                if not self.random_crop:
                    src = transform.resize(src, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True)
                src = self._min_max_scaling(src)
                src_list.append(src)

                tgt = self.target[i]
                tgt = transform.downscale_local_mean(tgt, (self.downscale, self.downscale))
                if not self.random_crop:
                    tgt = transform.resize(tgt, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True)
                if not self.binary_target:
                    tgt = self._min_max_scaling(tgt)
                tgt_list.append(tgt)

            if self.random_crop:
                if src.shape[0] == self.shape[0]:
                    x_rand = 0
                if src.shape[1] == self.shape[1]:
                    y_rand = 0
                if src.shape[0] > self.shape[0]:
                    x_rand = np.random.randint(src.shape[0] - self.shape[0])
                if src.shape[1] > self.shape[1]:
                    y_rand = np.random.randint(src.shape[1] - self.shape[1])
                if src.shape[0] < self.shape[0] or src.shape[1] < self.shape[1]:
                    raise ValueError('Patch shape larger than (downscaled) source shape')

            for i in range(self.shape[2]):
                if self.random_crop:
                    src = src_list[i]
                    tgt = tgt_list[i]
                    src_crop = src[x_rand:self.shape[0]+x_rand, y_rand:self.shape[1]+y_rand]
                    tgt_crop = tgt[x_rand:self.shape[0]+x_rand, y_rand:self.shape[1]+y_rand]
                else:
                    src_crop = src_list[i]
                    tgt_crop = tgt_list[i]

                source_batch[batch,:,:,i,0] = src_crop
                target_batch[batch,:,:,i,0] = tgt_crop

        if self.augment:
            # On-the-fly data augmentation
            source_batch, target_batch = self.augment_volume(source_batch, target_batch)

            # Data augmentation by reversing stack
            if np.random.random() > 0.5:
                source_batch, target_batch = source_batch[::-1], target_batch[::-1]

            # Data augmentation by elastic deformation
            if np.random.random() > 0.5 and self.deform_augment:
                source_batch, target_batch = self.deform_volume(source_batch, target_batch)

            if not self.binary_target:
                target_batch = self._min_max_scaling(target_batch)

            return self._min_max_scaling(source_batch), target_batch

        else:
            return source_batch, target_batch

    def on_epoch_end(self):
        # Validation split performed here
        self.batch_list = []
        # Create batch_list of all combinations of tifffile and stack position
        if self.dir_flag:
            for i in range(len(self.source_dir_list)):
                num_of_pages = tifffile.imread(self.source_dir_list[i]).shape[0]
                if self.is_val:
                    start_page = num_of_pages-math.floor(self.val_split*num_of_pages)
                    for j in range(start_page, num_of_pages-self.shape[2]):
                      self.batch_list.append([i, j])
                else:
                    last_page = math.floor((1-self.val_split)*num_of_pages)
                    for j in range(last_page-self.shape[2]):
                        self.batch_list.append([i, j])
        else:
            num_of_pages = self.source.shape[0]
            if self.is_val:
                start_page = num_of_pages-math.floor(self.val_split*num_of_pages)
                for j in range(start_page, num_of_pages-self.shape[2]):
                    self.batch_list.append([0, j])

            else:
                last_page = math.floor((1-self.val_split)*num_of_pages)
                for j in range(last_page-self.shape[2]):
                    self.batch_list.append([0, j])

        if self.is_val and (len(self.batch_list) <= 0):
            raise ValueError('validation_split too small! Increase val_split or decrease z-depth')
        random.shuffle(self.batch_list)

    def _min_max_scaling(self, data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data)

        return n/d

    def class_weights(self):
        ones = 0
        pixels = 0

        if self.dir_flag:
            for i in range(len(self.target_dir_list)):
                tgt = tifffile.imread(self.target_dir_list[i]).astype(bool)
                ones += np.sum(tgt)
                pixels += tgt.shape[0]*tgt.shape[1]*tgt.shape[2]
        else:
          ones = np.sum(self.target)
          pixels = self.target.shape[0]*self.target.shape[1]*self.target.shape[2]
        p_ones = ones/pixels
        p_zeros = 1-p_ones

        # Return swapped probability to increase weight of unlikely class
        return p_ones, p_zeros

    def deform_volume(self, src_vol, tgt_vol):
        [src_dfrm, tgt_dfrm] = elasticdeform.deform_random_grid([src_vol, tgt_vol],
                                                                axis=(1, 2, 3),
                                                                sigma=self.deform_sigma,
                                                                points=self.deform_points,
                                                                order=self.deform_order)
       # if self.binary_target:
           # tgt_dfrm = tgt_dfrm > 0.1

        return self._min_max_scaling(src_dfrm), tgt_dfrm

    def augment_volume(self, src_vol, tgt_vol):

        src_vol_aug = np.empty(src_vol.shape)
        tgt_vol_aug = np.empty(tgt_vol.shape)

        for i in range(src_vol.shape[3]):
            src_aug_z, tgt_aug_z = self.seq(images=src_vol[:,:,:,i,0].astype('float16'),
                                                                      segmentation_maps=np.expand_dims(tgt_vol[:,:,:,i,0].astype(bool), axis=-1))
            src_vol_aug[:,:,:,i,0] = src_aug_z
            tgt_vol_aug[:,:,:,i,0] = np.squeeze(tgt_aug_z)
        return self._min_max_scaling(src_vol_aug), tgt_vol_aug

    def sample_augmentation(self, idx):
        src, tgt = self.__getitem__(idx)

        src_aug, tgt_aug = self.augment_volume(src, tgt)

        if self.deform_augment:
            src_aug, tgt_aug = self.deform_volume(src_aug, tgt_aug)

        return src_aug, tgt_aug

# Define custom loss and dice coefficient
def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)

    return (2.*intersection)/(K.sum(y_true_f*y_true_f)+K.sum(y_pred_f*y_pred_f)+eps)

def weighted_binary_crossentropy(zero_weight, one_weight):
    def _weighted_binary_crossentropy(y_true, y_pred):
        binary_crossentropy = K.binary_crossentropy(y_true, y_pred)

        weight_vector = y_true*one_weight+(1.-y_true)*zero_weight
        weighted_binary_crossentropy = weight_vector*binary_crossentropy

        return K.mean(weighted_binary_crossentropy)

    return _weighted_binary_crossentropy

# Custom callback showing sample prediction
class SampleImageCallback(Callback):

    def __init__(self, model, sample_data, model_path, save=False):
        self.model = model
        self.sample_data = sample_data
        self.model_path = model_path
        self.save = save

    #def on_epoch_end(self, epoch, logs={}):
        #sample_predict = self.model.predict_on_batch(self.sample_data)

        #f=plt.figure(figsize=(16,8))
        #plt.subplot(1,2,1)
        #plt.imshow(self.sample_data[0,:,:,0,0], interpolation='nearest', cmap='gray')
        #plt.title('Sample source')
        #plt.axis('off');

        #plt.subplot(1,2,2)
        #plt.imshow(sample_predict[0,:,:,0,0], interpolation='nearest', cmap='magma')
        #plt.title('Predicted target')
        #plt.axis('off');

        #plt.show()

        if self.save:
            plt.savefig(self.model_path + '/epoch_' + str(epoch+1) + '.png')


# Define Unet3D class
class Unet3D:

    def __init__(self,
                 shape=(256,256,16,1)):
        if isinstance(shape, str):
            shape = eval(shape)

        self.shape = shape

        input_tensor = Input(self.shape, name='input')

        self.model = self.unet_3D(input_tensor)

    def down_block_3D(self, input_tensor, filters):
        x = Conv3D(filters=filters, kernel_size=(3,3,3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv3D(filters=filters*2, kernel_size=(3,3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def up_block_3D(self, input_tensor, concat_layer, filters):
        x = Conv3DTranspose(filters, kernel_size=(2,2,2), strides=(2,2,2))(input_tensor)

        x = Concatenate()([x, concat_layer])

        x = Conv3D(filters=filters, kernel_size=(3,3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv3D(filters=filters*2, kernel_size=(3,3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def unet_3D(self, input_tensor, filters=32):
        d1 = self.down_block_3D(input_tensor, filters=filters)
        p1 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), data_format='channels_last')(d1)
        d2 = self.down_block_3D(p1, filters=filters*2)
        p2 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), data_format='channels_last')(d2)
        d3 = self.down_block_3D(p2, filters=filters*4)
        p3 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), data_format='channels_last')(d3)

        d4 = self.down_block_3D(p3, filters=filters*8)

        u1 = self.up_block_3D(d4, d3, filters=filters*4)
        u2 = self.up_block_3D(u1, d2, filters=filters*2)
        u3 = self.up_block_3D(u2, d1, filters=filters)

        output_tensor = Conv3D(filters=1, kernel_size=(1,1,1), activation='sigmoid')(u3)

        return Model(inputs=[input_tensor], outputs=[output_tensor])

    def summary(self):
        return self.model.summary()

    # Pass generators instead
    def train(self,
              epochs,
              batch_size,
              train_generator,
              val_generator,
              model_path,
              model_name,
              optimizer='adam',
              learning_rate=0.001,
              loss='weighted_binary_crossentropy',
              metrics='dice',
              ckpt_period=1,
              save_best_ckpt_only=False,
              ckpt_path=None, neptune_run = neptune_run):

        class_weight_zero, class_weight_one = train_generator.class_weights()

        if loss == 'weighted_binary_crossentropy':
            loss = weighted_binary_crossentropy(class_weight_zero, class_weight_one)

        if metrics == 'dice':
            metrics = dice_coefficient

        if optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=[metrics])

        if ckpt_path is not None:
            self.model.load_weights(ckpt_path)

        full_model_path = os.path.join(model_path, model_name)

        if not os.path.exists(full_model_path):
            os.makedirs(full_model_path)

        log_dir = full_model_path + '/Quality Control'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        ckpt_dir =  full_model_path + '/ckpt'

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        csv_out_name = log_dir + '/training_evaluation.csv'
        if ckpt_path is None:
            csv_logger = CSVLogger(csv_out_name)
        else:
            csv_logger = CSVLogger(csv_out_name, append=True)

        if save_best_ckpt_only:
            ckpt_name = ckpt_dir + '/' + model_name + '.hdf5'
        else:
            ckpt_name = ckpt_dir + '/' + model_name + '_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5'

        model_ckpt = ModelCheckpoint(ckpt_name,
                                     verbose=1,
                                     save_freq=ckpt_period,
                                     save_best_only=save_best_ckpt_only,
                                     save_weights_only=True)

        sample_batch, __ = val_generator.__getitem__(random.randint(0, len(val_generator)))
        sample_img = SampleImageCallback(self.model,
                                         sample_batch,
                                         model_path)

        history_callback = self.model.fit(train_generator,
                       validation_data=val_generator,
                       validation_steps=math.floor(len(val_generator)/batch_size),
                       epochs=epochs,
                       callbacks=[csv_logger,
                                  model_ckpt,
                                  sample_img])

        last_ckpt_name = ckpt_dir + '/' + model_name + '_last.hdf5'
        self.model.save_weights(last_ckpt_name)

        loss_history = []
        val_loss_history = []
        epoch_history = []
        dice_coefficient_history = []
        val_dice_coefficient_history = []


        # Open the CSV file in append mode 
        with open(csv_out_name, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                loss_history.append(float(row['loss']))
                val_loss_history.append(float(row['val_loss']))
                epoch_history.append(int(row['epoch']))
                dice_coefficient_history.append(float(row['dice_coefficient']))
                val_dice_coefficient_history.append(float(row['val_dice_coefficient']))


        # Log loss values to Neptune
        if neptune_run is not None:
            neptune_run['loss'].log(loss_history)
            neptune_run['val_loss'].log(val_loss_history)
            neptune_run['epoch'].log(epoch_history)
            neptune_run['dice_coefficient'].log(dice_coefficient_history)
            neptune_run['val_dice_coefficient'].log(val_dice_coefficient_history)

    def _min_max_scaling(self, data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data)

        return n/d

    def predict(self,
                input,
                ckpt_path,
                z_range=None,
                downscaling=None,
                true_patch_size=None):

        self.model.load_weights(ckpt_path)

        if isinstance(downscaling, str):
            downscaling = eval(downscaling)

        if math.isnan(downscaling):
            downscaling = None

        if isinstance(true_patch_size, str):
            true_patch_size = eval(true_patch_size)

        if not isinstance(true_patch_size, tuple):
            if math.isnan(true_patch_size):
                true_patch_size = None

        if isinstance(input, str):
            src_volume = tifffile.imread(input)
        elif isinstance(input, np.ndarray):
            src_volume = input
        else:
            raise TypeError('Input is not path or numpy array!')

        in_size = src_volume.shape

        if downscaling or true_patch_size is not None:
            x_scaling = 0
            y_scaling = 0

            if true_patch_size is not None:
                x_scaling += true_patch_size[0]/self.shape[0]
                y_scaling += true_patch_size[1]/self.shape[1]
            if downscaling is not None:
                x_scaling += downscaling
                y_scaling += downscaling

            src_list = []
            for i in range(src_volume.shape[0]):
                 src_list.append(transform.downscale_local_mean(src_volume[i], (int(x_scaling), int(y_scaling))))
            src_volume = np.array(src_list)

        if z_range is not None:
            src_volume = src_volume[z_range[0]:z_range[1]]

        src_volume = self._min_max_scaling(src_volume)

        src_array = np.zeros((1,
                              math.ceil(src_volume.shape[1]/self.shape[0])*self.shape[0],
                              math.ceil(src_volume.shape[2]/self.shape[1])*self.shape[1],
                              math.ceil(src_volume.shape[0]/self.shape[2])*self.shape[2],
                              self.shape[3]))

        for i in range(src_volume.shape[0]):
            src_array[0,:src_volume.shape[1],:src_volume.shape[2],i,0] = src_volume[i]

        pred_array = np.empty(src_array.shape)
        print(src_volume.dtype)
        for i in range(math.ceil(src_volume.shape[1]/self.shape[0])):
          for j in range(math.ceil(src_volume.shape[2]/self.shape[1])):
            for k in range(math.ceil(src_volume.shape[0]/self.shape[2])):
                pred_temp = self.model.predict(src_array[:,
                                                         i*self.shape[0]:i*self.shape[0]+self.shape[0],
                                                         j*self.shape[1]:j*self.shape[1]+self.shape[1],
                                                         k*self.shape[2]:k*self.shape[2]+self.shape[2]])
                pred_array[:,
                           i*self.shape[0]:i*self.shape[0]+self.shape[0],
                           j*self.shape[1]:j*self.shape[1]+self.shape[1],
                           k*self.shape[2]:k*self.shape[2]+self.shape[2]] = pred_temp

        pred_volume = np.rollaxis(np.squeeze(pred_array), -1)[:src_volume.shape[0],:src_volume.shape[1],:src_volume.shape[2]]

        if downscaling is not None:
            pred_list = []
            for i in range(pred_volume.shape[0]):
                 pred_list.append(transform.resize(pred_volume[i], (in_size[1], in_size[2]), preserve_range=True))
            pred_volume = np.array(pred_list)

        return pred_volume

def pdf_export(trained = False, augmentation = False, pretrained_model = False):
  class MyFPDF(FPDF, HTMLMixin):
    pass

  pdf = MyFPDF()
  pdf.add_page()
  pdf.set_right_margin(-1)
  pdf.set_font("Arial", size = 11, style='B')

  day = datetime.now()
  datetime_str = str(day)[0:10]

  Header = 'Training report for '+Network+' model ('+model_name+')\nDate: '+datetime_str
  pdf.multi_cell(180, 5, txt = Header, align = 'L')
  pdf.ln(1)

  # add another cell
  if trained:
    training_time = "Training time: "+str(hour)+ "hour(s) "+str(mins)+"min(s) "+str(round(sec))+"sec(s)"
    pdf.cell(190, 5, txt = training_time, ln = 1, align='L')
  pdf.ln(1)

  Header_2 = 'Information for your materials and methods:'
  pdf.cell(190, 5, txt=Header_2, ln=1, align='L')
  pdf.ln(1)

  all_packages = ''
  for requirement in freeze(local_only=True):
    all_packages = all_packages+requirement+', '
  #print(all_packages)

  #Main Packages
  main_packages = ''
  version_numbers = []
  for name in ['tensorflow','numpy','keras']:
    find_name=all_packages.find(name)
    main_packages = main_packages+all_packages[find_name:all_packages.find(',',find_name)]+', '
    #Version numbers only here:
    version_numbers.append(all_packages[find_name+len(name)+2:all_packages.find(',',find_name)])

  try:
    cuda_version = subprocess.run(["nvcc","--version"],stdout=subprocess.PIPE)
    cuda_version = cuda_version.stdout.decode('utf-8')
    cuda_version = cuda_version[cuda_version.find(', V')+3:-1]
  except:
    cuda_version = ' - No cuda found - '
  try:
    gpu_name = subprocess.run(["nvidia-smi"],stdout=subprocess.PIPE)
    gpu_name = gpu_name.stdout.decode('utf-8')
    gpu_name = gpu_name[gpu_name.find('Tesla'):gpu_name.find('Tesla')+10]
  except:
    gpu_name = ' - No GPU found - '
  #print(cuda_version[cuda_version.find(', V')+3:-1])
  #print(gpu_name)

  if os.path.isdir(training_source):
    shape = io.imread(training_source+'/'+os.listdir(training_source)[0]).shape
  elif os.path.isfile(training_source):
    shape = io.imread(training_source).shape
  else:
    print('Cannot read training data.')

  dataset_size = len(train_generator)

  text = 'The '+Network+' model was trained from scratch for '+str(number_of_epochs)+' epochs on '+str(dataset_size)+' paired image patches (image dimensions: '+str(shape)+', patch size: ('+str(patch_size)+') with a batch size of '+str(batch_size)+' and a '+loss_function+' loss function, using the '+Network+' ZeroCostDL4Mic notebook (v '+Notebook_version[0]+') (von Chamier & Laine et al., 2020). Key python packages used include tensorflow (v '+version_numbers[0]+'), keras (v '+version_numbers[2]+'), numpy (v '+version_numbers[1]+'), cuda (v '+cuda_version+'). The training was accelerated using a '+gpu_name+'GPU.'

  if pretrained_model:
    text = 'The '+Network+' model was trained for '+str(number_of_epochs)+' epochs on '+str(dataset_size)+' paired image patches (image dimensions: '+str(shape)+', patch_size: '+str(patch_size)+') with a batch size of '+str(batch_size)+' and a '+loss_function+' loss function, using the '+Network+' ZeroCostDL4Mic notebook (v '+Notebook_version[0]+') (von Chamier & Laine et al., 2020). The model was retrained from a pretrained model. Key python packages used include tensorflow (v '+version_numbers[0]+'), keras (v '+version_numbers[2]+'), numpy (v '+version_numbers[1]+'), cuda (v '+cuda_version+'). The training was accelerated using a '+gpu_name+'GPU.'

  pdf.set_font('')
  pdf.set_font_size(10.)
  pdf.multi_cell(190, 5, txt = text, align='L')
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(28, 5, txt='Augmentation: ', ln=0)
  pdf.set_font('')
  if augmentation:
    aug_text = 'The dataset was augmented by'
    if add_gaussian_blur == True:
      aug_text = aug_text+'\n- gaussian blur'
    if add_linear_contrast == True:
      aug_text = aug_text+'\n- linear contrast'
    if add_additive_gaussian_noise == True:
      aug_text = aug_text+'\n- additive gaussian noise'
    if augmenters != '':
      aug_text = aug_text+'\n- imgaug augmentations: '+augmenters
    if add_elastic_deform == True:
      aug_text = aug_text+'\n- elastic deformation'
  else:
    aug_text = 'No augmentation was used for training.'
  pdf.multi_cell(190, 5, txt=aug_text, align='L')
  pdf.ln(1)
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.cell(180, 5, txt = 'Parameters', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font_size(10.)
  if use_default_advanced_parameters:
    pdf.cell(200, 5, txt='Default Advanced Parameters were enabled')
  pdf.cell(200, 5, txt='The following parameters were used for training:')
  pdf.ln(1)
  html = """
  <table width=60% style="margin-left:0px;">
    <tr>
      <th width = 50% align="left">Parameter</th>
      <th width = 50% align="left">Value</th>
    </tr>
    <tr>
      <td width = 50%>number_of_epochs</td>
      <td width = 50%>{0}</td>
    </tr>
    <tr>
      <td width = 50%>batch_size</td>
      <td width = 50%>{1}</td>
    </tr>
    <tr>
      <td width = 50%>patch_size</td>
      <td width = 50%>{2}</td>
    </tr>
    <tr>
      <td width = 50%>image_pre_processing</td>
      <td width = 50%>{3}</td>
    </tr>
    <tr>
      <td width = 50%>validation_split_in_percent</td>
      <td width = 50%>{4}</td>
    </tr>
      <tr>
      <td width = 50%>downscaling_in_xy</td>
      <td width = 50%>{5}</td>
    </tr>
      <tr>
      <td width = 50%>binary_target</td>
      <td width = 50%>{6}</td>
    </tr>
    <tr>
      <td width = 50%>loss_function</td>
      <td width = 50%>{7}</td>
    </tr>
    <tr>
      <td width = 50%>metrics</td>
      <td width = 50%>{8}</td>
    </tr>
    <tr>
      <td width = 50%>optimizer</td>
      <td width = 50%>{9}</td>
    </tr>
    <tr>
      <td width = 50%>checkpointing_period</td>
      <td width = 50%>{10}</td>
    </tr>
    <tr>
      <td width = 50%>save_best_only</td>
      <td width = 50%>{11}</td>
    </tr>
    <tr>
      <td width = 50%>resume_training</td>
      <td width = 50%>{12}</td>
    </tr>
  </table>
  """.format(number_of_epochs,batch_size,str(patch_size[0])+'x'+str(patch_size[1])+'x'+str(patch_size[2]),image_pre_processing, validation_split_in_percent, downscaling_in_xy, str(binary_target), loss_function, metrics, optimizer, checkpointing_period, str(save_best_only), str(resume_training))
  pdf.write_html(html)

  #pdf.multi_cell(190, 5, txt = text_2, align='L')
  pdf.set_font("Arial", size = 11, style='B')
  pdf.ln(1)
  pdf.cell(190, 5, txt = 'Training Dataset', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(30, 5, txt= 'Training_source:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = training_source, align = 'L')
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(28, 5, txt= 'Training_target:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = training_target, align = 'L')
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.cell(21, 5, txt= 'Model Path:', align = 'L', ln=0)
  pdf.set_font('')
  pdf.multi_cell(170, 5, txt = model_path+'/'+model_name, align = 'L')
  pdf.ln(1)
  #pdf.cell(60, 5, txt = 'Example Training pair (single slice)', ln=1)
  #pdf.ln(1)
  #exp_size = io.imread(base_path + '/TrainingDataExample_Unet3D.png').shape
  #pdf.image(base_path + '/TrainingDataExample_Unet3D.png', x = 11, y = None, w = round(exp_size[1]/8), h = round(exp_size[0]/8))
  #pdf.ln(1)
  ##ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
  #pdf.multi_cell(190, 5, txt = ref_1, align='L')
  #pdf.ln(1)
  #ref_2 = '- Unet 3D: Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation." International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.'
  #pdf.multi_cell(190, 5, txt = ref_2, align='L')
  # if Use_Data_augmentation:
  #   ref_4 = '- Augmentor: Bloice, Marcus D., Christof Stocker, and Andreas Holzinger. "Augmentor: an image augmentation library for machine learning." arXiv preprint arXiv:1708.04680 (2017).'
  #   pdf.multi_cell(190, 5, txt = ref_4, align='L')
  #pdf.ln(3)
  #reminder = 'Important:\nRemember to perform the quality control step on all newly trained models\nPlease consider depositing your training dataset on Zenodo'
  #pdf.set_font('Arial', size = 11, style='B')
  #pdf.multi_cell(190, 5, txt=reminder, align='C')
  #pdf.ln(1)

  pdf.output(model_path+'/'+model_name+'/'+model_name+'_training_report.pdf')

  print('------------------------------')
  print('PDF report exported in '+model_path+'/'+model_name+'/')


def qc_pdf_export():
  class MyFPDF(FPDF, HTMLMixin):
    pass

  pdf = MyFPDF()
  pdf.add_page()
  pdf.set_right_margin(-1)
  pdf.set_font("Arial", size = 11, style='B')

  Network = 'U-Net 3D'

  day = datetime.now()
  datetime_str = str(day)[0:10]

  Header = 'Quality Control report for '+Network+' model ('+qc_model_name+')\nDate: '+datetime_str
  pdf.multi_cell(180, 5, txt = Header, align = 'L')
  pdf.ln(1)

  all_packages = ''
  for requirement in freeze(local_only=True):
    all_packages = all_packages+requirement+', '

  pdf.set_font('')
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.ln(2)
  pdf.cell(190, 5, txt = 'Loss curves', ln=1, align='L')
  pdf.ln(1)
  if os.path.exists(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/lossCurvePlots.png'):
    exp_size = io.imread(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/lossCurvePlots.png').shape
    pdf.image(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/lossCurvePlots.png', x = 11, y = None, w = round(exp_size[1]/8), h = round(exp_size[0]/8))
  else:
    pdf.set_font('')
    pdf.set_font('Arial', size=10)
    pdf.multi_cell(190, 5, txt='If you would like to see the evolution of the loss function during training please play the first cell of the QC section in the notebook.')
  pdf.ln(2)
  pdf.set_font('')
  pdf.set_font('Arial', size = 10, style = 'B')
  pdf.ln(3)
  pdf.cell(80, 5, txt = 'Example Quality Control Visualisation', ln=1)
  pdf.ln(1)
  exp_size = io.imread(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/QC_example_data.png').shape
  pdf.image(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/QC_example_data.png', x = 5, y = None, w = round(exp_size[1]/12), h = round(exp_size[0]/8))
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font('Arial', size = 11, style = 'B')
  pdf.ln(1)
  pdf.cell(180, 5, txt = 'IoU threshold optimisation', align='L', ln=1)
  pdf.set_font('')
  pdf.set_font_size(10.)
  pdf.ln(1)
  pdf.cell(120, 5, txt='Highest IoU is {:.4f} with a threshold of {}'.format(best_iou, best_thresh), align='L', ln=1)
  pdf.ln(2)
  exp_size = io.imread(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/QC_IoU_analysis.png').shape
  pdf.image(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/QC_IoU_analysis.png', x=16, y=None, w = round(exp_size[1]/6), h = round(exp_size[0]/6))
  pdf.ln(1)
  pdf.set_font('')
  pdf.set_font_size(10.)
  ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
  pdf.multi_cell(190, 5, txt = ref_1, align='L')
  pdf.ln(1)
  ref_2 = '- Unet 3D: Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation." International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.'
  pdf.multi_cell(190, 5, txt = ref_2, align='L')
  pdf.ln(1)

  pdf.ln(3)
  reminder = 'To find the parameters and other information about how this model was trained, go to the training_report.pdf of this model which should be in the folder of the same name.'

  pdf.set_font('Arial', size = 11, style='B')
  pdf.multi_cell(190, 5, txt=reminder, align='C')
  pdf.ln(1)

  pdf.output(os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/'+qc_model_name+'_QC_report.pdf')

  print('------------------------------')
  print('QC PDF report exported in '+os.path.join(qc_model_path,qc_model_name,'Quality Control')+'/')


# -------------- Other definitions -----------
W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
prediction_prefix = 'Predicted_'


print('-------------------')
print('U-Net 3D and dependencies installed.')

# Colors for the warning messages
class bcolors:
  WARNING = '\033[31m'
  NORMAL = '\033[0m'  # white (normal)


# Check if this is the latest version of the notebook
# Latest_notebook_version = pd.read_csv("https://raw.githubusercontent.com/HenriquesLab/ZeroCostDL4Mic/master/Colab_notebooks/Latest_ZeroCostDL4Mic_Release.csv")

# if Notebook_version == list(Latest_notebook_version.columns):
#   print("This notebook is up-to-date.")

# if not Notebook_version == list(Latest_notebook_version.columns):
#   print(bcolors.WARNING +"A new version of this notebook has been released. We recommend that you download it at https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki")

All_notebook_versions = pd.read_csv("https://raw.githubusercontent.com/HenriquesLab/ZeroCostDL4Mic/master/Colab_notebooks/Latest_Notebook_versions.csv", dtype=str)
print('Notebook version: '+Notebook_version)
Latest_Notebook_version = All_notebook_versions[All_notebook_versions["Notebook"] == Network]['Version'].iloc[0]
print('Latest notebook version: '+Latest_Notebook_version)
if Notebook_version == Latest_Notebook_version:
  print("This notebook is up-to-date.")
else:
  print(bcolors.WARNING +"A new version of this notebook has been released. We recommend that you download it at https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki")


# Build requirements file for local run
#after = [str(m) for m in sys.modules]
#build_requirements_file(before, after)

if tf.test.gpu_device_name()=='':
  print('You do not have GPU access.')
  print('Did you change your runtime?')
  print('If the runtime setting is correct then Google did not allocate a GPU for your session')
  print('Expect slow performance. To access GPU try reconnecting later')

else:
  print('You have GPU access')


#using the argparse for options to add flags 

parser = argparse.ArgumentParser(description = "train")

parser.add_argument("--number_of_epochs",nargs="?", default=10,  type=int, required=False)

parser.add_argument("--model_name",nargs="?", default='model1', type=str, required=False)

parser.add_argument("--learning_rate",nargs="?", default=0.001, type=float, required=False)

parser.add_argument("--training_source", type=str, required=False)

parser.add_argument("--training_target", type=str, required=False)

parser.add_argument("--model_path", type=str, required=False)

parser.add_argument("--testing_source", type=str, required=False)

parser.add_argument("--testing_target", type=str, required=False)

parser.add_argument("--source_path", type=str, required=False)

parser.add_argument("--output_directory", type=str, required=False)

args = parser.parse_args()


#@markdown ###Path to training data:
training_source = args.training_source #@param {type:"string"}
training_target = args.training_target #@param {type:"string"}

#@markdown ---

#@markdown ###Model name and path to model folder:
model_name = args.model_name #@param {type:"string"}
model_path = args.model_path #@param {type:"string"}

full_model_path = os.path.join(model_path, model_name)

#@markdown ---

#@markdown ###Training parameters
number_of_epochs =   args.number_of_epochs #@param {type:"number"}

#@markdown ###Default advanced parameters
use_default_advanced_parameters = False #@param {type:"boolean"}

#@markdown <font size = 3>If not, please change:

batch_size =  10#@param {type:"number"}
patch_size = (64,64,8) #@param {type:"number"} # in pixels
training_shape = patch_size + (1,)
image_pre_processing = 'resize to patch_size' #@param ["randomly crop to patch_size", "resize to patch_size"]

validation_split_in_percent = 30 #@param{type:"number"}
downscaling_in_xy =  1#@param {type:"number"} # in pixels

binary_target = True #@param {type:"boolean"}

loss_function = 'weighted_binary_crossentropy' #@param ["weighted_binary_crossentropy", "binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "mean_squared_error", "mean_absolute_error"]

metrics = 'dice' #@param ["dice", "accuracy"]

optimizer = 'adam' #@param ["adam", "sgd", "rmsprop"]

learning_rate = 0.001 #@param{type:"number"}

if image_pre_processing == "randomly crop to patch_size":
    random_crop = True
else:
    random_crop = False

if use_default_advanced_parameters:
    print("Default advanced parameters enabled")
    batch_size = 3
    training_shape = (256,256,8,1)
    validation_split_in_percent = 20
    downscaling_in_xy = 1
    random_crop = True
    binary_target = True
    loss_function = 'weighted_binary_crossentropy'
    metrics = 'dice'
    optimizer = 'adam'
    learning_rate = 0.001
#@markdown ###Checkpointing parameters
#checkpointing_period = 1 #@param {type:"number"}
checkpointing_period = "epoch"
#@markdown  <font size = 3>If chosen, only the best checkpoint is saved. Otherwise a checkpoint is saved every epoch:
save_best_only = False #@param {type:"boolean"}

#@markdown ###Resume training
#@markdown <font size = 3>Choose if training was interrupted:
resume_training = False #@param {type:"boolean"}

#@markdown ###Transfer learning
#@markdown <font size = 3>For transfer learning, do not select resume_training and specify a checkpoint_path below.

#@markdown <font size = 3> - If the model is already downloaded or is locally available, please specify the path to the .h5 file.

#@markdown <font size = 3> - To use a model from the BioImage Model Zoo, write the model ID. For example: 10.5281/zenodo.5749843

pretrained_model_choice = "Model_from_file" #@param ["Model_from_file", "bioimageio_model"]
checkpoint_path = "" #@param {type:"string"}
model_id = "" #@param {type:"string"}
# --------------------- Load the model from a bioimageio model (can be path on drive or url / doi) ---
if pretrained_model_choice == "bioimageio_model":
  from bioimageio.core import load_raw_resource_description
  from zipfile import ZipFile
  import requests

  model_spec = load_raw_resource_description(model_id)
  if "keras_hdf5" not in model_spec.weights:
    print("Invalid bioimageio model")
  else:
    url = model_spec.weights["keras_hdf5"].source
    r = requests.get(url, allow_redirects=True)
    open("keras_model.h5", 'wb').write(r.content)
    checkpoint_path = "keras_model.h5"

if resume_training and checkpoint_path != "":
    print('If resume_training is True while checkpoint_path is specified, resume_training will be set to False!')
    resume_training = False

# Retrieve last checkpoint
if resume_training:
    try:
      ckpt_dir_list = glob(full_model_path + '/ckpt/*')
      ckpt_dir_list.sort()
      last_ckpt_path = ckpt_dir_list[-1]
      print('Training will resume from checkpoint:', os.path.basename(last_ckpt_path))
    except IndexError:
      last_ckpt_path=None
      print('CheckpointError: No previous checkpoints were found, training from scratch.')
elif not resume_training and checkpoint_path != "":
    last_ckpt_path = checkpoint_path
    assert os.path.isfile(last_ckpt_path), 'checkpoint_path does not exist!'
else:
    last_ckpt_path=None

# Instantiate Unet3D
model = Unet3D(shape=training_shape)

#here we check that no model with the same name already exist
if not resume_training and os.path.exists(full_model_path):
    print(bcolors.WARNING+'The model folder already exists and will be overwritten.'+bcolors.NORMAL)
    # print('!! WARNING: Folder already exists and will be overwritten !!')
    # shutil.rmtree(full_model_path)

# if not os.path.exists(full_model_path):
#     os.makedirs(full_model_path)

# Show sample image
if os.path.isdir(training_source):
    training_source_sample = sorted(glob(os.path.join(training_source, '*')))[0]
    training_target_sample = sorted(glob(os.path.join(training_target, '*')))[0]
else:
    training_source_sample = training_source
    training_target_sample = training_target

print(training_source)
print(training_source_sample)

src_sample = tifffile.imread(training_source_sample)
src_sample = model._min_max_scaling(src_sample)
if binary_target:
    tgt_sample = tifffile.imread(training_target_sample).astype(bool)
else:
    tgt_sample = tifffile.imread(training_target_sample)

src_down = transform.downscale_local_mean(src_sample[0], (downscaling_in_xy, downscaling_in_xy))
tgt_down = transform.downscale_local_mean(tgt_sample[0], (downscaling_in_xy, downscaling_in_xy))

if random_crop:
    true_patch_size = None

    if src_down.shape[0] == training_shape[0]:
      x_rand = 0
    if src_down.shape[1] == training_shape[1]:
      y_rand = 0
    if src_down.shape[0] > training_shape[0]:
      x_rand = np.random.randint(src_down.shape[0] - training_shape[0])
    if src_down.shape[1] > training_shape[1]:
      y_rand = np.random.randint(src_down.shape[1] - training_shape[1])
    if src_down.shape[0] < training_shape[0] or src_down.shape[1] < training_shape[1]:
      raise ValueError('Patch shape larger than (downscaled) source shape')
else:
    true_patch_size = src_down.shape

def scroll_in_z(z):
    src_down = transform.downscale_local_mean(src_sample[z-1], (downscaling_in_xy,downscaling_in_xy))
    tgt_down = transform.downscale_local_mean(tgt_sample[z-1], (downscaling_in_xy,downscaling_in_xy))
    if random_crop:
        src_slice = src_down[x_rand:training_shape[0]+x_rand, y_rand:training_shape[1]+y_rand]
        tgt_slice = tgt_down[x_rand:training_shape[0]+x_rand, y_rand:training_shape[1]+y_rand]
    else:

        src_slice = transform.resize(src_down, (training_shape[0], training_shape[1]), mode='constant', preserve_range=True)
        tgt_slice = transform.resize(tgt_down, (training_shape[0], training_shape[1]), mode='constant', preserve_range=True)

    #f=plt.figure(figsize=(16,8))
    #plt.subplot(1,2,1)
    #plt.imshow(src_slice, cmap='gray')
    #plt.title('Training source (z = ' + str(z) + ')', fontsize=15)
    #plt.axis('off')

    #plt.subplot(1,2,2)
    #plt.imshow(tgt_slice, cmap='magma')
    #plt.title('Training target (z = ' + str(z) + ')', fontsize=15)
    #plt.axis('off')
    #plt.savefig(base_path + '/TrainingDataExample_Unet3D.png',bbox_inches='tight',pad_inches=0)
    #plt.close()

#print('This is what the training images will look like with the chosen settings')
#interact(scroll_in_z, z=widgets.IntSlider(min=1, max=src_sample.shape[0], step=1, value=0));
#plt.show()
#Create a copy of an example slice and close the display.
#scroll_in_z(z=int(src_sample.shape[0]/2))
# If you close the display, then the users can't interactively inspect the data
# plt.close()

# Save model parameters
params =  {'training_source': training_source,
           'training_target': training_target,
           'model_name': model_name,
           'model_path': model_path,
           'number_of_epochs': number_of_epochs,
           'batch_size': batch_size,
           'training_shape': training_shape,
           'downscaling': downscaling_in_xy,
           'true_patch_size': true_patch_size,
           'val_split': validation_split_in_percent/100,
           'random_crop': random_crop}

neptune_run['parameters'] = params

params_df = pd.DataFrame.from_dict(params, orient='index')

# apply_data_augmentation = False
# pdf_export(augmentation = apply_data_augmentation, pretrained_model = resume_training)


#@markdown ##**Augmentation options**

#@markdown ###Data augmentation

apply_data_augmentation = True #@param {type:"boolean"}

# List of augmentations
augmentations = []

#@markdown ###Gaussian blur
add_gaussian_blur = True #@param {type:"boolean"}
gaussian_sigma =   0.7#@param {type:"number"}
gaussian_frequency = 0.5 #@param {type:"number"}

if add_gaussian_blur:
    augmentations.append(iaa.Sometimes(gaussian_frequency, iaa.GaussianBlur(sigma=(0, gaussian_sigma))))

#@markdown ###Linear contrast
add_linear_contrast = True #@param {type:"boolean"}
contrast_min =  0.4 #@param {type:"number"}
contrast_max =   1.6#@param {type:"number"}
contrast_frequency = 0.5 #@param {type:"number"}

if add_linear_contrast:
    augmentations.append(iaa.Sometimes(contrast_frequency, iaa.LinearContrast((contrast_min, contrast_max))))

#@markdown ###Additive Gaussian noise
add_additive_gaussian_noise = True #@param {type:"boolean"}
scale_min =  0 #@param {type:"number"}
scale_max =  0.05 #@param {type:"number"}
noise_frequency = 0.5 #@param {type:"number"}

if add_additive_gaussian_noise:
    augmentations.append(iaa.Sometimes(noise_frequency, iaa.AdditiveGaussianNoise(scale=(scale_min, scale_max))))

#@markdown ###Add custom augmenters
add_custom_augmenters = False #@param {type:"boolean"}
augmenters = "" #@param {type:"string"}

if add_custom_augmenters:

    augmenter_params = "" #@param {type:"string"}

    augmenter_frequency = "" #@param {type:"string"}

    aug_lst = augmenters.split(';')
    aug_params_lst = augmenter_params.split(';')
    aug_freq_lst = augmenter_frequency.split(';')

    assert len(aug_lst) == len(aug_params_lst) and len(aug_lst) == len(aug_freq_lst), 'The number of arguments in augmenters, augmenter_params and augmenter_frequency are not the same!'

    for __, (aug, param, freq) in enumerate(zip(aug_lst, aug_params_lst, aug_freq_lst)):
        aug, param, freq = aug.strip(), param.strip(), freq.strip()
        aug_func = iaa.Sometimes(eval(freq), getattr(iaa, aug)(eval(param)))
        augmentations.append(aug_func)

#@markdown ###Elastic deformations
add_elastic_deform = True  #@param {type:"boolean"}
sigma =  2#@param {type:"number"}
points =  2#@param {type:"number"}
order =  1#@param {type:"number"}

if add_elastic_deform:
    deform_params = (sigma, points, order)
else:
    deform_params = None

train_generator = MultiPageTiffGenerator(training_source,
                                         training_target,
                                         batch_size=batch_size,
                                         shape=training_shape,
                                         augment=apply_data_augmentation,
                                         augmentations=augmentations,
                                         deform_augment=add_elastic_deform,
                                         deform_augmentation_params=deform_params,
                                         val_split=validation_split_in_percent/100,
                                         random_crop=random_crop,
                                         downscale=downscaling_in_xy,
                                         binary_target=binary_target)

val_generator = MultiPageTiffGenerator(training_source,
                                       training_target,
                                       batch_size=batch_size,
                                       shape=training_shape,
                                       val_split=validation_split_in_percent/100,
                                       is_val=True,
                                       random_crop=random_crop,
                                       downscale=downscaling_in_xy,
                                       binary_target=binary_target)


if apply_data_augmentation:
  print('Data augmentation enabled.')
  sample_src_aug, sample_tgt_aug = train_generator.sample_augmentation(random.randint(0, len(train_generator)))

  #def scroll_in_z(z):
      #f=plt.figure(figsize=(16,8))
      #plt.subplot(1,2,1)
      #plt.imshow(sample_src_aug[0,:,:,z-1,0], cmap='gray')
      #plt.title('Sample augmented source (z = ' + str(z) + ')', fontsize=15)
      #plt.axis('off')

      #plt.subplot(1,2,2)
      #plt.imshow(sample_tgt_aug[0,:,:,z-1,0], cmap='magma')
      #plt.title('Sample training target (z = ' + str(z) + ')', fontsize=15)
      #plt.axis('off')

  #print('This is what the augmented training images will look like with the chosen settings')
  #interact(scroll_in_z, z=widgets.IntSlider(min=1, max=sample_src_aug.shape[3], step=1, value=0));

else:
  print('Data augmentation disabled.')


#@markdown ## Show model summary
model.summary()

#@markdown ##Start training

#here we check that no model with the same name already exist, if so delete
if not resume_training and os.path.exists(full_model_path):
    shutil.rmtree(full_model_path)
    print(bcolors.WARNING+'!! WARNING: Folder already exists and has been overwritten !!'+bcolors.NORMAL)

if not os.path.exists(full_model_path):
    os.makedirs(full_model_path)

pdf_export(augmentation = apply_data_augmentation, pretrained_model = resume_training)

# Save file
params_df.to_csv(os.path.join(full_model_path, 'params.csv'))

start = time.time()
# Start Training
model.train(epochs=number_of_epochs,
            batch_size=batch_size,
            train_generator=train_generator,
            val_generator=val_generator,
            model_path=model_path,
            model_name=model_name,
            loss=loss_function,
            metrics=metrics,
            optimizer=optimizer,
            learning_rate=learning_rate,
            ckpt_period=checkpointing_period,
            save_best_ckpt_only=save_best_only,
            ckpt_path=last_ckpt_path, neptune_run=neptune_run)
#print(_weighted_binary_crossentropy)
print('Training successfully completed!')
dt = time.time() - start
mins, sec = divmod(dt, 60)
hour, mins = divmod(mins, 60)
print("Time elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)")

#Create a pdf document with training summary

#pdf_export(trained = True, augmentation = apply_data_augmentation, pretrained_model = resume_training)


#@markdown ##Compare prediction and ground-truth on testing data

#@markdown <font size = 4>Provide an unseen annotated dataset to determine the performance of the model:

testing_source = args.testing_source #@param{type:"string"}
testing_target = args.testing_target #@param{type:"string"}

qc_dir = full_model_path + '/Quality Control'
predict_dir = qc_dir + '/Prediction'
if os.path.exists(predict_dir):
    shutil.rmtree(predict_dir)

os.makedirs(predict_dir)

# predict_dir + '/' +
predict_path = os.path.splitext(os.path.basename(testing_source))[0] + '_prediction.tif'

def last_chars(x):
    return(x[-11:])

try:
    ckpt_dir_list = glob(full_model_path + '/ckpt/*')
    ckpt_dir_list.sort(key=last_chars)
    last_ckpt_path = ckpt_dir_list[0]
    print('Predicting from checkpoint:', os.path.basename(last_ckpt_path))
except IndexError:
    raise CheckpointError('No previous checkpoints were found, please retrain model.')

# Load parameters
params = pd.read_csv(os.path.join(full_model_path, 'params.csv'), names=['val'], header=0, index_col=0)

model = Unet3D(shape=params.loc['training_shape', 'val'])

prediction = model.predict(testing_source, last_ckpt_path,
                           downscaling=params.loc['downscaling', 'val'],
                           true_patch_size=params.loc['true_patch_size', 'val'])

tifffile.imwrite(predict_path, prediction.astype('float32'), imagej=True)

print('Predicted images!')

#qc_metrics_path = full_model_path + '/Quality Control/QC_metrics_' + qc_model_name + '.csv'

test_target = tifffile.imread(testing_target)
test_source = tifffile.imread(testing_source)
test_prediction = tifffile.imread(predict_path)


#@markdown ##Calculate Intersection over Union and best threshold
prediction = tifffile.imread(predict_path)
prediction = np.interp(prediction, (prediction.min(), prediction.max()), (0, 255))

target = tifffile.imread(testing_target).astype(bool)

def iou_vs_threshold(prediction, target):
    threshold_list = []
    IoU_scores_list = []

    for threshold in range(0,256):
        mask = prediction > threshold

        intersection = np.logical_and(target, mask)
        union = np.logical_or(target, mask)
        iou_score = np.sum(intersection) / np.sum(union)

        threshold_list.append(threshold)
        IoU_scores_list.append(iou_score)

    return threshold_list, IoU_scores_list

threshold_list, IoU_scores_list = iou_vs_threshold(prediction, target)
thresh_arr = np.array(list(zip(threshold_list, IoU_scores_list)))
best_thresh = int(np.where(thresh_arr == np.max(thresh_arr[:,1]))[0])
best_iou = IoU_scores_list[best_thresh]

print('Highest IoU is {:.4f} with a threshold of {}'.format(best_iou, best_thresh))


#@markdown ### Provide the path to your dataset and to the folder where the predictions are saved, then run the cell to predict outputs from your unseen images.

source_path = args.source_path #@param {type:"string"}
output_directory = args.output_directory #@param {type:"string"}

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_path = os.path.join(output_directory, os.path.splitext(os.path.basename(source_path))[0] + '_predicted.tif')
#@markdown ###Prediction parameters:

binary_target = True #@param {type:"boolean"}

save_probability_map = False #@param {type:"boolean"}

#@markdown <font size = 3>Determine best threshold in Section 5.2.

use_calculated_threshold = True #@param {type:"boolean"}
threshold =  200#@param {type:"number"}

# Tifffile library issues means that images cannot be appended to
#@markdown <font size = 3>Choose if prediction file exceeds 4GB or if input file is very large (above 2GB). Image volume saved as BigTIFF.
big_tiff = False #@param {type:"boolean"}

#@markdown <font size = 3>Reduce `prediction_depth` if runtime runs out of memory during prediction. Only relevant if prediction saved as BigTIFF

prediction_depth =  32#@param {type:"number"}

#@markdown ###Model to be evaluated
#@markdown <font size = 3>If left blank, the latest model defined in Section 5 will be evaluated


# Load parameters
params = pd.read_csv(os.path.join(full_model_path, 'params.csv'), names=['val'], header=0, index_col=0)
model = Unet3D(shape=params.loc['training_shape', 'val'])

if use_calculated_threshold:
    threshold = best_thresh

def last_chars(x):
    return(x[-11:])

try:
    ckpt_dir_list = glob(full_model_path + '/ckpt/*')
    ckpt_dir_list.sort(key=last_chars)
    last_ckpt_path = ckpt_dir_list[0]
    print('Predicting from checkpoint:', os.path.basename(last_ckpt_path))
except IndexError:
    raise CheckpointError('No previous checkpoints were found, please retrain model.')

src = tifffile.imread(source_path)

if src.nbytes >= 4e9:
    big_tiff = True
    print('The source file exceeds 4GB in memory, prediction will be saved as BigTIFF!')

if binary_target:
    if not big_tiff:
        prediction = model.predict(src, last_ckpt_path, downscaling=params.loc['downscaling', 'val'], true_patch_size=params.loc['true_patch_size', 'val'])
        prediction = np.interp(prediction, (prediction.min(), prediction.max()), (0, 255))
        prediction = (prediction > threshold).astype('float32')
        prediction = prediction.reshape(prediction.shape[0], 1, prediction.shape[1], prediction.shape[2])

        tifffile.imwrite(output_path, prediction, imagej=True)

    else:
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            for i in tqdm(range(0, src.shape[0], prediction_depth)):
                prediction = model.predict(src, last_ckpt_path, z_range=(i,i+prediction_depth), downscaling=params.loc['downscaling', 'val'], true_patch_size=params.loc['true_patch_size', 'val'])
                prediction = np.interp(prediction, (prediction.min(), prediction.max()), (0, 255))
                prediction = (prediction > threshold).astype('float32')

                for j in range(prediction.shape[0]):
                    tif.write(prediction[j])

if not binary_target or save_probability_map:
    if not binary_target:
        prob_map_path = output_path
    else:
        prob_map_path = os.path.splitext(output_path)[0] + '_prob_map.tif'

    if not big_tiff:
        prediction = model.predict(src, last_ckpt_path, downscaling=params.loc['downscaling', 'val'], true_patch_size=params.loc['true_patch_size', 'val'])
        prediction = np.interp(prediction, (prediction.min(), prediction.max()), (0, 255))
        prediction = prediction.reshape(prediction.shape[0], 1, prediction.shape[1], prediction.shape[2])
        tifffile.imwrite(prob_map_path, prediction.astype('float32'), imagej=True)

    else:
        with tifffile.TiffWriter(prob_map_path, bigtiff=True) as tif:
            for i in tqdm(range(0, src.shape[0], prediction_depth)):
                prediction = model.predict(src, last_ckpt_path, z_range=(i,i+prediction_depth), downscaling=params.loc['downscaling', 'val'], true_patch_size=params.loc['true_patch_size', 'val'])
                prediction = np.interp(prediction, (prediction.min(), prediction.max()), (0, 255))

                for j in range(prediction.shape[0]):
                    tif.write(prediction[j])

print('Predictions saved as', output_path)

src_volume = tifffile.imread(source_path)
pred_volume = tifffile.imread(output_path)
