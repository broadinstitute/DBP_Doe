from __future__ import absolute_import, division, print_function, unicode_literals

#@markdown ##Load key 3D U-Net dependencies and instantiate network
Notebook_version = '2.2.1'
Network = 'U-Net (3D)'

from builtins import any as b_any


import utils 
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

from neptune.types import File
from neptune.integrations.tensorflow_keras import NeptuneCallback

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

#EXPERIMENT_DESCRIPTION = 'Track the training details'

#neptune_run = neptune.init_run(
    #project="BroadImagingPlatform/DBP-Doe",
    #api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMWU0MmFiZS0yZGVkLTQwMGItYTczNC0yNzdiNTljMTExY2QifQ==")

#neptune_callback = NeptuneCallback(run=neptune_run)

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


All_notebook_versions = pd.read_csv("https://raw.githubusercontent.com/HenriquesLab/ZeroCostDL4Mic/master/Colab_notebooks/Latest_Notebook_versions.csv", dtype=str)
print('Notebook version: '+Notebook_version)
Latest_Notebook_version = All_notebook_versions[All_notebook_versions["Notebook"] == Network]['Version'].iloc[0]
print('Latest notebook version: '+Latest_Notebook_version)
if Notebook_version == Latest_Notebook_version:
  print("This notebook is up-to-date.")
else:
  print(bcolors.WARNING +"A new version of this notebook has been released. We recommend that you download it at https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki")


if tf.test.gpu_device_name()=='':
  print('You do not have GPU access.')
  print('Did you change your runtime?')
  print('If the runtime setting is correct then Google did not allocate a GPU for your session')
  print('Expect slow performance. To access GPU try reconnecting later')

else:
  print('You have GPU access')

# Defining the parameters 

#Using the argparse for options to add flags 


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

parser.add_argument("--batch_size", type=int, default=1, required=False)

parser.add_argument("--loss_function", type=str, default='weighted_binary_crossentropy', required=False)

parser.add_argument("--optimizer", type=str, default='adam', required=False)

args = parser.parse_args()

#@markdown ###Path to training data:
training_source = args.training_source #@param {type:"string"}
training_target = args.training_target #@param {type:"string"}

#@markdown ---

#@markdown ###Model name and path to model folder:
model_name = args.model_name  #@param {type:"string"}
model_path = args.model_path #@param {type:"string"}

full_model_path = os.path.join(model_path, model_name)

#@markdown ---

#@markdown ###Training parameters
number_of_epochs =   args.number_of_epochs #@param {type:"number"}

#@markdown ###Default advanced parameters
use_default_advanced_parameters = False #@param {type:"boolean"}

#@markdown <font size = 3>If not, please change:

batch_size = args.batch_size #@param {type:"number"}
patch_size = (64,64,8) #@param {type:"number"} # in pixels
training_shape = patch_size + (1,)
image_pre_processing = 'randomly crop to patch_size' #@param ["randomly crop to patch_size", "resize to patch_size"]

validation_split_in_percent = 20 #@param{type:"number"}
downscaling_in_xy =  1#@param {type:"number"} # in pixels

binary_target = True #@param {type:"boolean"}

loss_function = args.loss_function #@param ["weighted_binary_crossentropy", "binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "mean_squared_error", "mean_absolute_error"]

metrics = 'dice' #@param ["dice", "accuracy"]

optimizer = args.optimizer #@param ["adam", "sgd", "rmsprop"]

learning_rate = args.learning_rate #@param{type:"number"}

if image_pre_processing == "randomly crop to patch_size":
    random_crop = True
else:
    random_crop = False

#if use_default_advanced_parameters:
    #print("Default advanced parameters enabled")
    #batch_size = 3
    #training_shape = (256,256,8,1)
    #validation_split_in_percent = 50
    #downscaling_in_xy = 1
    #random_crop = True
    #binary_target = True
    #loss_function = 'weighted_binary_crossentropy'
    #metrics = 'dice'
    #optimizer = 'adam'
    #learning_rate = 0.001
#@markdown ###Checkpointing parameters
checkpointing_period = 1 #@param {type:"number"}
#checkpointing_period = "epoch"
#@markdown  <font size = 3>If chosen, only the best checkpoint is saved. Otherwise a checkpoint is saved every epoch:
save_best_only = False #@param {type:"boolean"}

#@markdown ###Resume training
resume_training = False #@param {type:"boolean"}



pretrained_model_choice = "Model_from_file" #@param ["Model_from_file", "bioimageio_model"]
checkpoint_path = "" #@param {type:"string"}
model_id = "" #@param {type:"string"}


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
model = utils.Unet3D(shape=training_shape)

#here we check that no model with the same name already exist
if not resume_training and os.path.exists(full_model_path):
    print(bcolors.WARNING+'The model folder already exists and will be overwritten.'+bcolors.NORMAL)
    # print('!! WARNING: Folder already exists and will be overwritten !!')
    # shutil.rmtree(full_model_path)


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
           'random_crop': random_crop,
           'learning_rate':learning_rate,
           'patch_size':patch_size,
           'loss_function':loss_function,
           'optimizer':optimizer,
           'metrics':metrics}

utils.neptune_run['parameters'] = params

params_df = pd.DataFrame.from_dict(params, orient='index')

#@markdown ##**Augmentation options**

#@markdown ###Data augmentation

apply_data_augmentation = True #@param {type:"boolean"}

# List of augmentations
augmentations = []

#@markdown ###Gaussian blur
add_gaussian_blur = True #@param {type:"boolean"}
gaussian_sigma =   0.7 #@param {type:"number"}
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
add_elastic_deform = False  #@param {type:"boolean"}
sigma =  2#@param {type:"number"}
points =  2#@param {type:"number"}
order =  1#@param {type:"number"}

if add_elastic_deform:
    deform_params = (sigma, points, order)
else:
    deform_params = None

train_generator = utils.MultiPageTiffGenerator(training_source,
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

val_generator = utils.MultiPageTiffGenerator(training_source,
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

  
else:
  print('Data augmentation disabled.')


#@markdown ## Show model summary
#model.summary()

__import__("IPython").embed()
#@markdown ##Start training

#here we check that no model with the same name already exist, if so delete
if not resume_training and os.path.exists(full_model_path):
    shutil.rmtree(full_model_path)
    print(bcolors.WARNING+'!! WARNING: Folder already exists and has been overwritten !!'+bcolors.NORMAL)

if not os.path.exists(full_model_path):
    os.makedirs(full_model_path)


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
            ckpt_path=last_ckpt_path, neptune_run=utils.neptune_run)
#print(_weighted_binary_crossentropy)
print('Training successfully completed!')
dt = time.time() - start
mins, sec = divmod(dt, 60)
hour, mins = divmod(mins, 60)
print("Time elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)")


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

#model = Unet3D(shape=params.loc['training_shape', 'val'])

prediction = model.predict(testing_source, last_ckpt_path,
                           downscaling=params.loc['downscaling', 'val'],
                           true_patch_size=params.loc['true_patch_size', 'val'])

tifffile.imwrite(predict_path, prediction.astype('float32'), imagej=True)

print('Predicted images!')

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
model = utils.Unet3D(shape=params.loc['training_shape', 'val'])

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


#Making intensity projection of the predicted image to log it in neptune 

source_max_proj = np.max(src_volume, axis=0)
pred_max_proj = np.max(pred_volume, axis=0)


#Writing the maximum intensity projected files to a folder
source_max_path = os.path.join(output_directory, os.path.splitext(os.path.basename(source_path))[0] + '_source_max.tif')
pred_max_path = os.path.join(output_directory, os.path.splitext(os.path.basename(source_path))[0] + '_predicted_max.tif')

tifffile.imwrite(source_max_path, source_max_proj.astype('uint8'), imagej=True)
tifffile.imwrite(pred_max_path, pred_max_proj.astype('float32'), imagej=True)

source_max = tifffile.imread(source_max_path)
pred_max = tifffile.imread(pred_max_path)

utils.neptune_run['source_max'].upload(File.as_image(source_max))
utils.neptune_run['pred_max'].upload(File.as_image(pred_max))


