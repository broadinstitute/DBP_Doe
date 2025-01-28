# Run this cell to execute the code
from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime
import ipywidgets as widgets
from IPython.display import Markdown, display, clear_output
from matplotlib import pyplot as plt
import yaml as yaml_library
import os

ipywidgets_edit_yaml_config_path = os.path.join(os.getcwd(), 'results', 'widget_prev_settings.yaml')

def ipywidgets_edit_yaml(yaml_path, key, value):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config_data = yaml_library.safe_load(f)
    else:
        config_data = {}
    config_data[key] = value
    with open(yaml_path, 'w') as new_f:
        yaml_library.safe_dump(config_data, new_f, width=10e10, default_flow_style=False, allow_unicode=True)

def ipywidgets_read_yaml(yaml_path, key):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config_data = yaml_library.safe_load(f)
        value = config_data.get(key, '')
        return value
    else:
        return ''

internal_aux_initial_time=datetime.now()
print('Runnning...')
print('--------------------------------------')

Notebook_version = '2.2.1'
Network = 'U-Net (3D)'

from builtins import any as b_any

def get_requirements_path():
    # Store requirements file in 'contents' directory 
    current_dir = os.getcwd()
    dir_count = current_dir.count('/') - 1
    path = '../' * (dir_count) + 'requirements.txt'
    return path

def filter_files(file_list, filter_list):
    filtered_list = []
    for fname in file_list:
        if b_any(fname.split('==')[0] in s for s in filter_list):
            filtered_list.append(fname)
    return filtered_list

def build_requirements_file(before, after):
    path = get_requirements_path()

    # Exporting requirements.txt for local run
    #!pip freeze > $path

    # Get minimum requirements file
    df = pd.read_csv(path)
    mod_list = [m.split('.')[0] for m in after if not m in before]
    req_list_temp = df.values.tolist()
    req_list = [x[0] for x in req_list_temp]

    # Replace with package name and handle cases where import name is different to module name
    mod_name_list = [['sklearn', 'scikit-learn'], ['skimage', 'scikit-image']]
    mod_replace_list = [[x[1] for x in mod_name_list] if s in [x[0] for x in mod_name_list] else s for s in mod_list] 
    filtered_list = filter_files(req_list, mod_replace_list)

    file=open(path,'w')
    for item in filtered_list:
        file.writelines(item)

    file.close()

import sys
before = [str(m) for m in sys.modules]

#Put the imported code and libraries here

try:
    import elasticdeform
except:
    import elasticdeform

try:
    import tifffile
except:
    import tifffile

try:
    import imgaug.augmenters as iaa
except:
    import imgaug.augmenters as iaa

import os
import csv
import random
import h5py
import imageio
import math
import shutil

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
from bioimageio.core.build_spec import build_model, add_weights
from bioimageio.core.resource_tests import test_model
from bioimageio.core.weight_converter.keras import convert_weights_to_tensorflow_saved_model_bundle

print("Dependencies installed and imported.")

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
        if self.binary_target:
            tgt_dfrm = tgt_dfrm > 0.1
        
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

    def on_epoch_end(self, epoch, logs={}):
        sample_predict = self.model.predict_on_batch(self.sample_data)

        f=plt.figure(figsize=(16,8))
        plt.subplot(1,2,1)
        plt.imshow(self.sample_data[0,:,:,0,0], interpolation='nearest', cmap='gray')
        plt.title('Sample source')
        plt.axis('off');

        plt.subplot(1,2,2)
        plt.imshow(sample_predict[0,:,:,0,0], interpolation='nearest', cmap='magma')
        plt.title('Predicted target')
        plt.axis('off');

        plt.show()

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
              ckpt_path=None):

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

        self.model.fit(train_generator,
                       validation_data=val_generator,
                       validation_steps=math.floor(len(val_generator)/batch_size),
                       epochs=epochs,
                       callbacks=[csv_logger,
                                  model_ckpt,
                                  sample_img])

        last_ckpt_name = ckpt_dir + '/' + model_name + '_last.hdf5'
        self.model.save_weights(last_ckpt_name)

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
  pdf.cell(60, 5, txt = 'Example Training pair (single slice)', ln=1)
  pdf.ln(1)
  exp_size = io.imread(base_path + '/TrainingDataExample_Unet3D.png').shape
  pdf.image(base_path + '/TrainingDataExample_Unet3D.png', x = 11, y = None, w = round(exp_size[1]/8), h = round(exp_size[0]/8))
  pdf.ln(1)
  ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
  pdf.multi_cell(190, 5, txt = ref_1, align='L')
  pdf.ln(1)
  ref_2 = '- Unet 3D: Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation." International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.'
  pdf.multi_cell(190, 5, txt = ref_2, align='L')
  # if Use_Data_augmentation:
  #   ref_4 = '- Augmentor: Bloice, Marcus D., Christof Stocker, and Andreas Holzinger. "Augmentor: an image augmentation library for machine learning." arXiv preprint arXiv:1708.04680 (2017).'
  #   pdf.multi_cell(190, 5, txt = ref_4, align='L')
  pdf.ln(3)
  reminder = 'Important:\nRemember to perform the quality control step on all newly trained models\nPlease consider depositing your training dataset on Zenodo'
  pdf.set_font('Arial', size = 11, style='B')
  pdf.multi_cell(190, 5, txt=reminder, align='C')
  pdf.ln(1)

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
after = [str(m) for m in sys.modules]
build_requirements_file(before, after)
#!pip3 freeze > requirements.txt
print('--------------------------------------')
print(f'Finnished. Duration: {datetime.now() - internal_aux_initial_time}')


# Run this cell to visualize the parameters and click the button to execute the code
internal_aux_initial_time=datetime.now()
clear_output()

display(Markdown("### Path to training data:"))
widget_training_source = widgets.Text(value="", style={'description_width': 'initial'}, description="training_source:")
display(widget_training_source)
widget_training_target = widgets.Text(value="", style={'description_width': 'initial'}, description="training_target:")
display(widget_training_target)
display(Markdown(" ---"))
display(Markdown("### Model name and path to model folder:"))
widget_model_name = widgets.Text(value="", style={'description_width': 'initial'}, description="model_name:")
display(widget_model_name)
widget_model_path = widgets.Text(value="", style={'description_width': 'initial'}, description="model_path:")
display(widget_model_path)
display(Markdown(" ---"))
display(Markdown("### Training parameters"))
widget_number_of_epochs = widgets.IntText(value=2, style={'description_width': 'initial'}, description="number_of_epochs:")
display(widget_number_of_epochs)
display(Markdown("### Default advanced parameters"))
widget_use_default_advanced_parameters = widgets.Checkbox(value=False, style={'description_width': 'initial'}, description="use_default_advanced_parameters:")
display(widget_use_default_advanced_parameters)
display(Markdown(" <font size = 3>If not, please change:"))
widget_batch_size = widgets.IntText(value=1, style={'description_width': 'initial'}, description="batch_size:")
display(widget_batch_size)
widget_patch_size = widgets.Text(value="""(512,512,8)""", style={'description_width': 'initial'}, description="patch_size:")
display(widget_patch_size)
widget_image_pre_processing = widgets.Dropdown(options=["randomly crop to patch_size", "resize to patch_size"], value="randomly crop to patch_size", style={'description_width': 'initial'}, description="image_pre_processing:")
display(widget_image_pre_processing)
widget_validation_split_in_percent = widgets.IntText(value=20, style={'description_width': 'initial'}, description="validation_split_in_percent:")
display(widget_validation_split_in_percent)
widget_downscaling_in_xy = widgets.Text(value="""1""", style={'description_width': 'initial'}, description="downscaling_in_xy:")
display(widget_downscaling_in_xy)
widget_binary_target = widgets.Checkbox(value=True, style={'description_width': 'initial'}, description="binary_target:")
display(widget_binary_target)
widget_loss_function = widgets.Dropdown(options=["weighted_binary_crossentropy", "binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "mean_squared_error", "mean_absolute_error"], value="weighted_binary_crossentropy", style={'description_width': 'initial'}, description="loss_function:")
display(widget_loss_function)
widget_metrics = widgets.Dropdown(options=["dice", "accuracy"], value="dice", style={'description_width': 'initial'}, description="metrics:")
display(widget_metrics)
widget_optimizer = widgets.Dropdown(options=["adam", "sgd", "rmsprop"], value="adam", style={'description_width': 'initial'}, description="optimizer:")
display(widget_optimizer)
widget_learning_rate = widgets.FloatText(value=0.001, style={'description_width': 'initial'}, description="learning_rate:")
display(widget_learning_rate)
display(Markdown("### Checkpointing parameters"))
widget_checkpointing_period = widgets.IntText(value=1, style={'description_width': 'initial'}, description="checkpointing_period:")
display(widget_checkpointing_period)
display(Markdown(" <font size = 3>If chosen, only the best checkpoint is saved. Otherwise a checkpoint is saved every epoch:"))
widget_save_best_only = widgets.Checkbox(value=False, style={'description_width': 'initial'}, description="save_best_only:")
display(widget_save_best_only)
display(Markdown("### Resume training"))
display(Markdown(" <font size = 3>Choose if training was interrupted:"))
widget_resume_training = widgets.Checkbox(value=False, style={'description_width': 'initial'}, description="resume_training:")
display(widget_resume_training)
display(Markdown("### Transfer learning"))
display(Markdown(" <font size = 3>For transfer learning, do not select resume_training and specify a checkpoint_path below. "))
display(Markdown(" <font size = 3> - If the model is already downloaded or is locally available, please specify the path to the .h5 file. "))
display(Markdown(" <font size = 3> - To use a model from the BioImage Model Zoo, write the model ID. For example: 10.5281/zenodo.5749843"))
widget_pretrained_model_choice = widgets.Dropdown(options=["Model_from_file", "bioimageio_model"], value="Model_from_file", style={'description_width': 'initial'}, description="pretrained_model_choice:")
display(widget_pretrained_model_choice)
widget_checkpoint_path = widgets.Text(value="", style={'description_width': 'initial'}, description="checkpoint_path:")
display(widget_checkpoint_path)
widget_model_id = widgets.Text(value="", style={'description_width': 'initial'}, description="model_id:")
display(widget_model_id)

def function_8(output_widget):
  output_widget.clear_output()
  with output_widget:
    global training_source
    global training_target
    global model_name
    global model_path
    global number_of_epochs
    global use_default_advanced_parameters
    global batch_size
    global patch_size
    global image_pre_processing
    global validation_split_in_percent
    global downscaling_in_xy
    global binary_target
    global loss_function
    global metrics
    global optimizer
    global learning_rate
    global checkpointing_period
    global save_best_only
    global resume_training
    global pretrained_model_choice
    global checkpoint_path
    global model_id

    global full_model_path
    global training_shape
    global random_crop
    global random_crop
    global batch_size
    global training_shape
    global validation_split_in_percent
    global downscaling_in_xy
    global random_crop
    global binary_target
    global loss_function
    global metrics
    global optimizer
    global learning_rate
    global checkpointing_period
    global model_spec
    global url
    global r
    global checkpoint_path
    global resume_training
    global ckpt_dir_list
    global last_ckpt_path
    global last_ckpt_path
    global last_ckpt_path
    global last_ckpt_path
    global model
    global training_source_sample
    global training_target_sample
    global training_source_sample
    global training_target_sample
    global src_sample
    global src_sample
    global tgt_sample
    global tgt_sample
    global src_down
    global tgt_down
    global true_patch_size
    global x_rand
    global y_rand
    global x_rand
    global y_rand
    global true_patch_size
    global src_down
    global tgt_down
    global src_slice
    global tgt_slice
    global src_slice
    global tgt_slice
    global f
    global params
    global params_df

    global scroll_in_z

    training_source = widget_training_source.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_training_source', widget_training_source.value)
    training_target = widget_training_target.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_training_target', widget_training_target.value)
    
    model_name = widget_model_name.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_model_name', widget_model_name.value)
    model_path = widget_model_path.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_model_path', widget_model_path.value)
    
    full_model_path = os.path.join(model_path, model_name)
    
    number_of_epochs = widget_number_of_epochs.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_number_of_epochs', widget_number_of_epochs.value)
    
    use_default_advanced_parameters = widget_use_default_advanced_parameters.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_use_default_advanced_parameters', widget_use_default_advanced_parameters.value)
    
    batch_size = widget_batch_size.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_batch_size', widget_batch_size.value)
    patch_size = eval(widget_patch_size.value)
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_patch_size', eval(widget_patch_size.value))
    training_shape = patch_size + (1,)
    image_pre_processing = widget_image_pre_processing.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_image_pre_processing', widget_image_pre_processing.value)
    
    validation_split_in_percent = widget_validation_split_in_percent.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_validation_split_in_percent', widget_validation_split_in_percent.value)
    downscaling_in_xy = eval(widget_downscaling_in_xy.value)
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_downscaling_in_xy', eval(widget_downscaling_in_xy.value))
    
    binary_target = widget_binary_target.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_binary_target', widget_binary_target.value)
    
    loss_function = widget_loss_function.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_loss_function', widget_loss_function.value)
    
    metrics = widget_metrics.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_metrics', widget_metrics.value)
    
    optimizer = widget_optimizer.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_optimizer', widget_optimizer.value)
    
    learning_rate = widget_learning_rate.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_learning_rate', widget_learning_rate.value)
    
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
    checkpointing_period = widget_checkpointing_period.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_checkpointing_period', widget_checkpointing_period.value)
    checkpointing_period = "epoch"
    save_best_only = widget_save_best_only.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_save_best_only', widget_save_best_only.value)
    
    resume_training = widget_resume_training.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_resume_training', widget_resume_training.value)
    
    pretrained_model_choice = widget_pretrained_model_choice.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_pretrained_model_choice', widget_pretrained_model_choice.value)
    checkpoint_path = widget_checkpoint_path.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_checkpoint_path', widget_checkpoint_path.value)
    model_id = widget_model_id.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_8_model_id', widget_model_id.value)
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
    
    src_sample = tifffile.imread(training_source_sample)
    src_sample = model._min_max_scaling(src_sample)
    if binary_target:
        tgt_sample = tifffile.imread(training_target_sample).astype(np.bool)
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
    
        f=plt.figure(figsize=(16,8))
        plt.subplot(1,2,1)
        plt.imshow(src_slice, cmap='gray')
        plt.title('Training source (z = ' + str(z) + ')', fontsize=15)
        plt.axis('off')
    
        plt.subplot(1,2,2)
        plt.imshow(tgt_slice, cmap='magma')
        plt.title('Training target (z = ' + str(z) + ')', fontsize=15)
        plt.axis('off')
        plt.savefig(base_path + '/TrainingDataExample_Unet3D.png',bbox_inches='tight',pad_inches=0)
        #plt.close()
    
    print('This is what the training images will look like with the chosen settings')
    interact(scroll_in_z, z=widgets.IntSlider(min=1, max=src_sample.shape[0], step=1, value=0));
    plt.show()
    #Create a copy of an example slice and close the display.
    scroll_in_z(z=int(src_sample.shape[0]/2))
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
    
    params_df = pd.DataFrame.from_dict(params, orient='index')
    
    # apply_data_augmentation = False
    # pdf_export(augmentation = apply_data_augmentation, pretrained_model = resume_training)
    
    plt.show()

def function_8_cache(output_widget):
    global training_source
    global training_target
    global model_name
    global model_path
    global number_of_epochs
    global use_default_advanced_parameters
    global batch_size
    global patch_size
    global image_pre_processing
    global validation_split_in_percent
    global downscaling_in_xy
    global binary_target
    global loss_function
    global metrics
    global optimizer
    global learning_rate
    global checkpointing_period
    global save_best_only
    global resume_training
    global pretrained_model_choice
    global checkpoint_path
    global model_id

    global full_model_path
    global training_shape
    global random_crop
    global random_crop
    global batch_size
    global training_shape
    global validation_split_in_percent
    global downscaling_in_xy
    global random_crop
    global binary_target
    global loss_function
    global metrics
    global optimizer
    global learning_rate
    global checkpointing_period
    global model_spec
    global url
    global r
    global checkpoint_path
    global resume_training
    global ckpt_dir_list
    global last_ckpt_path
    global last_ckpt_path
    global last_ckpt_path
    global last_ckpt_path
    global model
    global training_source_sample
    global training_target_sample
    global training_source_sample
    global training_target_sample
    global src_sample
    global src_sample
    global tgt_sample
    global tgt_sample
    global src_down
    global tgt_down
    global true_patch_size
    global x_rand
    global y_rand
    global x_rand
    global y_rand
    global true_patch_size
    global src_down
    global tgt_down
    global src_slice
    global tgt_slice
    global src_slice
    global tgt_slice
    global f
    global params
    global params_df

    global scroll_in_z

    cache_training_source = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_training_source')
    if cache_training_source != '':
        widget_training_source.value = cache_training_source
    
    cache_training_target = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_training_target')
    if cache_training_target != '':
        widget_training_target.value = cache_training_target
    
    cache_model_name = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_model_name')
    if cache_model_name != '':
        widget_model_name.value = cache_model_name
    
    cache_model_path = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_model_path')
    if cache_model_path != '':
        widget_model_path.value = cache_model_path
    
    cache_number_of_epochs = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_number_of_epochs')
    if cache_number_of_epochs != '':
        widget_number_of_epochs.value = cache_number_of_epochs
    
    cache_use_default_advanced_parameters = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_use_default_advanced_parameters')
    if cache_use_default_advanced_parameters != '':
        widget_use_default_advanced_parameters.value = cache_use_default_advanced_parameters
    
    cache_batch_size = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_batch_size')
    if cache_batch_size != '':
        widget_batch_size.value = cache_batch_size
    
    cache_patch_size = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_patch_size')
    if cache_patch_size != '':
        widget_patch_size.value = cache_patch_size
    
    cache_image_pre_processing = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_image_pre_processing')
    if cache_image_pre_processing != '':
        widget_image_pre_processing.value = cache_image_pre_processing
    
    cache_validation_split_in_percent = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_validation_split_in_percent')
    if cache_validation_split_in_percent != '':
        widget_validation_split_in_percent.value = cache_validation_split_in_percent
    
    cache_downscaling_in_xy = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_downscaling_in_xy')
    if cache_downscaling_in_xy != '':
        widget_downscaling_in_xy.value = cache_downscaling_in_xy
    
    cache_binary_target = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_binary_target')
    if cache_binary_target != '':
        widget_binary_target.value = cache_binary_target
    
    cache_loss_function = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_loss_function')
    if cache_loss_function != '':
        widget_loss_function.value = cache_loss_function
    
    cache_metrics = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_metrics')
    if cache_metrics != '':
        widget_metrics.value = cache_metrics
    
    cache_optimizer = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_optimizer')
    if cache_optimizer != '':
        widget_optimizer.value = cache_optimizer
    
    cache_learning_rate = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_learning_rate')
    if cache_learning_rate != '':
        widget_learning_rate.value = cache_learning_rate
    
    cache_checkpointing_period = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_checkpointing_period')
    if cache_checkpointing_period != '':
        widget_checkpointing_period.value = cache_checkpointing_period
    
    cache_save_best_only = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_save_best_only')
    if cache_save_best_only != '':
        widget_save_best_only.value = cache_save_best_only
    
    cache_resume_training = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_resume_training')
    if cache_resume_training != '':
        widget_resume_training.value = cache_resume_training
    
    cache_pretrained_model_choice = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_pretrained_model_choice')
    if cache_pretrained_model_choice != '':
        widget_pretrained_model_choice.value = cache_pretrained_model_choice
    
    cache_checkpoint_path = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_checkpoint_path')
    if cache_checkpoint_path != '':
        widget_checkpoint_path.value = cache_checkpoint_path
    
    cache_model_id = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_8_model_id')
    if cache_model_id != '':
        widget_model_id.value = cache_model_id
    
button_function_8 = widgets.Button(description='Load and run')
cache_button_function_8 = widgets.Button(description='Load prev. settings')
output_function_8 = widgets.Output()
display(widgets.HBox((button_function_8, cache_button_function_8)), output_function_8)
def aux_function_8(_):
  return function_8(output_function_8)

def aux_function_8_cache(_):
  return function_8_cache(output_function_8)

button_function_8.on_click(aux_function_8)
cache_button_function_8.on_click(aux_function_8_cache)
print('--------------------------------------------------------------')
print('^ Introduce the arguments and click "Load and run". ^')
print('^ Or first click "Load prev. settings" if any previous ^')
print('^ settings have been saved and then click "Load and run". ^')


# Run this cell to visualize the parameters and click the button to execute the code
internal_aux_initial_time=datetime.now()
clear_output()

display(Markdown("## **Augmentation options**"))
display(Markdown("### Data augmentation"))
widget_apply_data_augmentation = widgets.Checkbox(value=True, style={'description_width': 'initial'}, description="apply_data_augmentation:")
display(widget_apply_data_augmentation)
display(Markdown("### Gaussian blur"))
widget_add_gaussian_blur = widgets.Checkbox(value=True, style={'description_width': 'initial'}, description="add_gaussian_blur:")
display(widget_add_gaussian_blur)
widget_gaussian_sigma = widgets.FloatText(value=0.7, style={'description_width': 'initial'}, description="gaussian_sigma:")
display(widget_gaussian_sigma)
widget_gaussian_frequency = widgets.FloatText(value=0.5, style={'description_width': 'initial'}, description="gaussian_frequency:")
display(widget_gaussian_frequency)
display(Markdown("### Linear contrast"))
widget_add_linear_contrast = widgets.Checkbox(value=True, style={'description_width': 'initial'}, description="add_linear_contrast:")
display(widget_add_linear_contrast)
widget_contrast_min = widgets.FloatText(value=0.4, style={'description_width': 'initial'}, description="contrast_min:")
display(widget_contrast_min)
widget_contrast_max = widgets.FloatText(value=1.6, style={'description_width': 'initial'}, description="contrast_max:")
display(widget_contrast_max)
widget_contrast_frequency = widgets.FloatText(value=0.5, style={'description_width': 'initial'}, description="contrast_frequency:")
display(widget_contrast_frequency)
display(Markdown("### Additive Gaussian noise"))
widget_add_additive_gaussian_noise = widgets.Checkbox(value=True, style={'description_width': 'initial'}, description="add_additive_gaussian_noise:")
display(widget_add_additive_gaussian_noise)
widget_scale_min = widgets.IntText(value=0, style={'description_width': 'initial'}, description="scale_min:")
display(widget_scale_min)
widget_scale_max = widgets.FloatText(value=0.05, style={'description_width': 'initial'}, description="scale_max:")
display(widget_scale_max)
widget_noise_frequency = widgets.FloatText(value=0.5, style={'description_width': 'initial'}, description="noise_frequency:")
display(widget_noise_frequency)
display(Markdown("### Add custom augmenters"))
widget_add_custom_augmenters = widgets.Checkbox(value=False, style={'description_width': 'initial'}, description="add_custom_augmenters:")
display(widget_add_custom_augmenters)
widget_augmenters = widgets.Text(value="", style={'description_width': 'initial'}, description="augmenters:")
display(widget_augmenters)
widget_augmenter_params = widgets.Text(value="", style={'description_width': 'initial'}, description="augmenter_params:")
display(widget_augmenter_params)
widget_augmenter_frequency = widgets.Text(value="", style={'description_width': 'initial'}, description="augmenter_frequency:")
display(widget_augmenter_frequency)
display(Markdown("### Elastic deformations"))
widget_add_elastic_deform = widgets.Checkbox(value=False, style={'description_width': 'initial'}, description="add_elastic_deform:")
display(widget_add_elastic_deform)
widget_sigma = widgets.IntText(value=2, style={'description_width': 'initial'}, description="sigma:")
display(widget_sigma)
widget_points = widgets.IntText(value=2, style={'description_width': 'initial'}, description="points:")
display(widget_points)
widget_order = widgets.IntText(value=1, style={'description_width': 'initial'}, description="order:")
display(widget_order)

def function_10(output_widget):
  output_widget.clear_output()
  with output_widget:
    global apply_data_augmentation
    global add_gaussian_blur
    global gaussian_sigma
    global gaussian_frequency
    global add_linear_contrast
    global contrast_min
    global contrast_max
    global contrast_frequency
    global add_additive_gaussian_noise
    global scale_min
    global scale_max
    global noise_frequency
    global add_custom_augmenters
    global augmenters
    global augmenter_params
    global augmenter_frequency
    global add_elastic_deform
    global sigma
    global points
    global order

    global augmentations
    global aug_lst
    global aug_params_lst
    global aug_freq_lst
    global aug
    global  param
    global  freq
    global aug_func
    global deform_params
    global deform_params
    global train_generator
    global batch_size
    global shape
    global augment
    global augmentations
    global deform_augment
    global deform_augmentation_params
    global val_split
    global random_crop
    global downscale
    global binary_target
    global val_generator
    global batch_size
    global shape
    global val_split
    global is_val
    global random_crop
    global downscale
    global binary_target
    global sample_src_aug
    global  sample_tgt_aug
    global f

    global scroll_in_z

    apply_data_augmentation = widget_apply_data_augmentation.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_apply_data_augmentation', widget_apply_data_augmentation.value)
    
    # List of augmentations
    augmentations = []
    
    add_gaussian_blur = widget_add_gaussian_blur.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_gaussian_blur', widget_add_gaussian_blur.value)
    gaussian_sigma = widget_gaussian_sigma.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_gaussian_sigma', widget_gaussian_sigma.value)
    gaussian_frequency = widget_gaussian_frequency.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_gaussian_frequency', widget_gaussian_frequency.value)
    
    if add_gaussian_blur:
        augmentations.append(iaa.Sometimes(gaussian_frequency, iaa.GaussianBlur(sigma=(0, gaussian_sigma))))
    
    add_linear_contrast = widget_add_linear_contrast.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_linear_contrast', widget_add_linear_contrast.value)
    contrast_min = widget_contrast_min.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_contrast_min', widget_contrast_min.value)
    contrast_max = widget_contrast_max.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_contrast_max', widget_contrast_max.value)
    contrast_frequency = widget_contrast_frequency.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_contrast_frequency', widget_contrast_frequency.value)
    
    if add_linear_contrast:
        augmentations.append(iaa.Sometimes(contrast_frequency, iaa.LinearContrast((contrast_min, contrast_max))))
    
    add_additive_gaussian_noise = widget_add_additive_gaussian_noise.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_additive_gaussian_noise', widget_add_additive_gaussian_noise.value)
    scale_min = widget_scale_min.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_scale_min', widget_scale_min.value)
    scale_max = widget_scale_max.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_scale_max', widget_scale_max.value)
    noise_frequency = widget_noise_frequency.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_noise_frequency', widget_noise_frequency.value)
    
    if add_additive_gaussian_noise:
        augmentations.append(iaa.Sometimes(noise_frequency, iaa.AdditiveGaussianNoise(scale=(scale_min, scale_max))))
    
    add_custom_augmenters = widget_add_custom_augmenters.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_custom_augmenters', widget_add_custom_augmenters.value)
    augmenters = widget_augmenters.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_augmenters', widget_augmenters.value)
    
    if add_custom_augmenters:
    
        augmenter_params = widget_augmenter_params.value
        ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_augmenter_params', widget_augmenter_params.value)
    
        augmenter_frequency = widget_augmenter_frequency.value
        ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_augmenter_frequency', widget_augmenter_frequency.value)
    
        aug_lst = augmenters.split(';')
        aug_params_lst = augmenter_params.split(';')
        aug_freq_lst = augmenter_frequency.split(';')
    
        assert len(aug_lst) == len(aug_params_lst) and len(aug_lst) == len(aug_freq_lst), 'The number of arguments in augmenters, augmenter_params and augmenter_frequency are not the same!'
    
        for __, (aug, param, freq) in enumerate(zip(aug_lst, aug_params_lst, aug_freq_lst)):
            aug, param, freq = aug.strip(), param.strip(), freq.strip() 
            aug_func = iaa.Sometimes(eval(freq), getattr(iaa, aug)(eval(param)))
            augmentations.append(aug_func)
    
    add_elastic_deform = widget_add_elastic_deform.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_elastic_deform', widget_add_elastic_deform.value)
    sigma = widget_sigma.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_sigma', widget_sigma.value)
    points = widget_points.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_points', widget_points.value)
    order = widget_order.value
    ipywidgets_edit_yaml(ipywidgets_edit_yaml_config_path, 'function_10_order', widget_order.value)
    
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
    
      def scroll_in_z(z):
          f=plt.figure(figsize=(16,8))
          plt.subplot(1,2,1)
          plt.imshow(sample_src_aug[0,:,:,z-1,0], cmap='gray')
          plt.title('Sample augmented source (z = ' + str(z) + ')', fontsize=15)
          plt.axis('off')
    
          plt.subplot(1,2,2)
          plt.imshow(sample_tgt_aug[0,:,:,z-1,0], cmap='magma')
          plt.title('Sample training target (z = ' + str(z) + ')', fontsize=15)
          plt.axis('off')
    
      print('This is what the augmented training images will look like with the chosen settings')
      interact(scroll_in_z, z=widgets.IntSlider(min=1, max=sample_src_aug.shape[3], step=1, value=0));
    
    else:
      print('Data augmentation disabled.')
    
    plt.show()

def function_10_cache(output_widget):
    global apply_data_augmentation
    global add_gaussian_blur
    global gaussian_sigma
    global gaussian_frequency
    global add_linear_contrast
    global contrast_min
    global contrast_max
    global contrast_frequency
    global add_additive_gaussian_noise
    global scale_min
    global scale_max
    global noise_frequency
    global add_custom_augmenters
    global augmenters
    global augmenter_params
    global augmenter_frequency
    global add_elastic_deform
    global sigma
    global points
    global order

    global augmentations
    global aug_lst
    global aug_params_lst
    global aug_freq_lst
    global aug
    global  param
    global  freq
    global aug_func
    global deform_params
    global deform_params
    global train_generator
    global batch_size
    global shape
    global augment
    global augmentations
    global deform_augment
    global deform_augmentation_params
    global val_split
    global random_crop
    global downscale
    global binary_target
    global val_generator
    global batch_size
    global shape
    global val_split
    global is_val
    global random_crop
    global downscale
    global binary_target
    global sample_src_aug
    global  sample_tgt_aug
    global f

    global scroll_in_z

    cache_apply_data_augmentation = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_apply_data_augmentation')
    if cache_apply_data_augmentation != '':
        widget_apply_data_augmentation.value = cache_apply_data_augmentation
    
    cache_add_gaussian_blur = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_gaussian_blur')
    if cache_add_gaussian_blur != '':
        widget_add_gaussian_blur.value = cache_add_gaussian_blur
    
    cache_gaussian_sigma = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_gaussian_sigma')
    if cache_gaussian_sigma != '':
        widget_gaussian_sigma.value = cache_gaussian_sigma
    
    cache_gaussian_frequency = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_gaussian_frequency')
    if cache_gaussian_frequency != '':
        widget_gaussian_frequency.value = cache_gaussian_frequency
    
    cache_add_linear_contrast = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_linear_contrast')
    if cache_add_linear_contrast != '':
        widget_add_linear_contrast.value = cache_add_linear_contrast
    
    cache_contrast_min = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_contrast_min')
    if cache_contrast_min != '':
        widget_contrast_min.value = cache_contrast_min
    
    cache_contrast_max = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_contrast_max')
    if cache_contrast_max != '':
        widget_contrast_max.value = cache_contrast_max
    
    cache_contrast_frequency = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_contrast_frequency')
    if cache_contrast_frequency != '':
        widget_contrast_frequency.value = cache_contrast_frequency
    
    cache_add_additive_gaussian_noise = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_additive_gaussian_noise')
    if cache_add_additive_gaussian_noise != '':
        widget_add_additive_gaussian_noise.value = cache_add_additive_gaussian_noise
    
    cache_scale_min = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_scale_min')
    if cache_scale_min != '':
        widget_scale_min.value = cache_scale_min
    
    cache_scale_max = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_scale_max')
    if cache_scale_max != '':
        widget_scale_max.value = cache_scale_max
    
    cache_noise_frequency = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_noise_frequency')
    if cache_noise_frequency != '':
        widget_noise_frequency.value = cache_noise_frequency
    
    cache_add_custom_augmenters = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_custom_augmenters')
    if cache_add_custom_augmenters != '':
        widget_add_custom_augmenters.value = cache_add_custom_augmenters
    
    cache_augmenters = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_augmenters')
    if cache_augmenters != '':
        widget_augmenters.value = cache_augmenters
    
    cache_augmenter_params = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_augmenter_params')
    if cache_augmenter_params != '':
        widget_augmenter_params.value = cache_augmenter_params
    
    cache_augmenter_frequency = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_augmenter_frequency')
    if cache_augmenter_frequency != '':
        widget_augmenter_frequency.value = cache_augmenter_frequency
    
    cache_add_elastic_deform = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_add_elastic_deform')
    if cache_add_elastic_deform != '':
        widget_add_elastic_deform.value = cache_add_elastic_deform
    
    cache_sigma = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_sigma')
    if cache_sigma != '':
        widget_sigma.value = cache_sigma
    
    cache_points = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_points')
    if cache_points != '':
        widget_points.value = cache_points
    
    cache_order = ipywidgets_read_yaml(ipywidgets_edit_yaml_config_path, 'function_10_order')
    if cache_order != '':
        widget_order.value = cache_order
    
button_function_10 = widgets.Button(description='Load and run')
cache_button_function_10 = widgets.Button(description='Load prev. settings')
output_function_10 = widgets.Output()
display(widgets.HBox((button_function_10, cache_button_function_10)), output_function_10)
def aux_function_10(_):
  return function_10(output_function_10)

def aux_function_10_cache(_):
  return function_10_cache(output_function_10)

button_function_10.on_click(aux_function_10)
cache_button_function_10.on_click(aux_function_10_cache)
print('--------------------------------------------------------------')
print('^ Introduce the arguments and click "Load and run". ^')
print('^ Or first click "Load prev. settings" if any previous ^')
print('^ settings have been saved and then click "Load and run". ^')


# Run this cell to execute the code
internal_aux_initial_time=datetime.now()
print('Runnning...')
print('--------------------------------------')
model.summary()
print('--------------------------------------')
print(f'Finnished. Duration: {datetime.now() - internal_aux_initial_time}')


# Run this cell to execute the code
internal_aux_initial_time=datetime.now()
print('Runnning...')
print('--------------------------------------')

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
            ckpt_path=last_ckpt_path)

print('Training successfully completed!')
dt = time.time() - start
mins, sec = divmod(dt, 60) 
hour, mins = divmod(mins, 60) 
print("Time elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)")

#Create a pdf document with training summary

pdf_export(trained = True, augmentation = apply_data_augmentation, pretrained_model = resume_training)
print('--------------------------------------')
print(f'Finnished. Duration: {datetime.now() - internal_aux_initial_time}')


