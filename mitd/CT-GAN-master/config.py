import os
import numpy as np
import tensorflow as tf

#consider your coordinate system, and x vs y

config = {}

# Data Location
config['healthy_scans_raw'] = "data/healthy_scans/" #path to directory where the healthy scans are. Filename is patient ID.
config['healthy_coords'] = "data/healthy_coords.csv" #path to csv where each row indicates where a healthy sample is (format: filename, x, y, z). 'fileneame' is the folder containing the dcm files of that scan or the mhd file name, slice is the z axis
config['healthy_samples'] = "data/healthy_samples.npy" #path to pickle dump of processed healthy samples for training.
config['unhealthy_scans_raw'] = "data/unhealthy_scans/" #path to directory where the unhealthy scans are
config['unhealthy_coords'] = "data/unhealthy_coords.csv" #path to csv where each row indicates where a healthy sample is (format: filename, x, y ,z)
config['unhealthy_samples'] = "data/unhealthy_samples.npy" #path to pickle dump of processed healthy samples for training.

config['traindata_coordSystem'] = "world" # the coord system used to note the locations of the evidence ('world' or 'vox'). vox is array index.

# Model & Progress Location
config['modelpath_inject'] = os.path.join("data","models","INJ") #path to save/load trained models and normalization parameters for injector
config['modelpath_remove'] = os.path.join("data","models","REM") #path to save/load trained models and normalization parameters for remover
config['progress'] = "images" #path to save snapshots of training progress

# tensorflow configuration
gpus = tf.config.list_physical_devices('GPU') 

if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


if len(gpus) > 0: #if there are GPUs avalaible...
    config['gpus'] = "0" #sets which GPU to use (use_CPU:"", use_GPU0:"0", etc...)
else:
    config['gpus'] = ""

# CT-GAN Configuration
config['cube_shape'] = np.array([32,32,32]) #z,y,x
config['mask_xlims'] = np.array([6,26])
config['mask_ylims'] = np.array([6,26])
config['mask_zlims'] = np.array([6,26])
config['copynoise'] = True #If true, the noise touch-up is copied onto the tampered region from a hardcoded coordinate. If false, gaussain interpolated noise is added instead

if config['mask_zlims'][1] > config['cube_shape'][0]:
    raise Exception('Out of bounds: cube mask is larger then cube on dimension z.')
if config['mask_ylims'][1] > config['cube_shape'][1]:
    raise Exception('Out of bounds: cube mask is larger then cube on dimension y.')
if config['mask_xlims'][1] > config['cube_shape'][2]:
    raise Exception('Out of bounds: cube mask is larger then cube on dimension x.')

# Make save directories
if not os.path.exists(config['modelpath_inject']):
    os.makedirs(config['modelpath_inject'])
if not os.path.exists(config['modelpath_remove']):
    os.makedirs(config['modelpath_remove'])
if not os.path.exists(config['progress']):
    os.makedirs(config['progress'])
