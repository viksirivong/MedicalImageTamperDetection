import csv
import numpy as np
import sys
import os
import pandas as pd
from scipy import ndimage

from utils.dicom_utils import load_dicom

DEFAULT_LOAD_PATH   = None
MAX_FILES_PROCESSED = 100       # limit on how many files to process.
DESIRED_DEPTH       = 64        # dimension for rescaling.
DESIRED_WIDTH       = 128       # dimension for rescaling.
DESIRED_HEIGHT      = 128       # dimension for rescaling.
MIN_HU              = -1024     # estimate of minimum Hounsfield unit of CT scan.
MAX_HU              = 400       # estimate of maximum Hounsfield unit of CT scan. HU >= 400 --> bone

def resize_volume(image):
    # Get current depth
    current_depth  = image.shape[0]
    current_width  = image.shape[1]
    current_height = image.shape[2]
    # compute dimensions.
    depth  = current_depth  / DESIRED_DEPTH
    width  = current_width  / DESIRED_WIDTH
    height = current_height / DESIRED_HEIGHT
    # compute depth factor.
    depth_factor  = 1 / depth
    width_factor  = 1 / width
    height_factor = 1 / height
    # resize across dimension of depth.
    image = ndimage.zoom(image, (depth_factor, height_factor, width_factor))
    return image

def normalize(image):
    image[image < MIN_HU] = MIN_HU
    image[image > MAX_HU] = MAX_HU
    image = (image - MIN_HU) / (MAX_HU - MIN_HU)
    image = image.astype("float32")
    return image

'''
    buildDataSet load_path=OPTIONAL save_path=OPTIONAL
        load_path is the directory containing a list of folders each with 2 scans.
        In each of these folders exists original_scan.npy and tampered_scan.npy
        This program will gather all of original_scan.npy and tampered_scan.npy  and write to data.npz.

        If no load_path is provided, the default one shall be used.
        If no save_path is provided, the default one is load_path.
'''
def main():
    # if no load path is provided, use the default one.
    load_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_LOAD_PATH
    # if no save path is provided, save_path will be load path.
    save_path = sys.argv[2] if len(sys.argv) == 3 else load_path 
    
    X     = []
    Y     = []
    count = 0
    files = os.listdir(load_path)

    for file in files:
        # stop if we processed as many files as desired.
        if count > MAX_FILES_PROCESSED:
            break
        # skip any file that isn't a folder containing the two CT scans.
        if not ('0' <= file.split('.')[-1][-1] <= '9'):
            print("Skipped file: {}".format(file))
            continue
        # if more than 2 files, it is .dcm 
        if len(os.listdir(file)) > 2:
            scan, spacing, orientation, origin, raw_slices = load_dicom(load_path + '\\' + file)
            X.append(scan)
            Y.append(1)
        else:
            # load original/tampered scans.
            original = np.load(load_path + "\\" + file + '\\original_scan.npy')
            tampered = np.load(load_path + "\\" + file + '\\tampered_scan.npy')
            # normalize original/tampered scans.
            original = normalize(original)
            tampered = normalize(tampered)
            # resize original/tampered scans for efficiency.
            original = resize_volume(original)
            tampered = resize_volume(tampered)
            # add both scans of type ndarray to list.
            X.append(original)
            X.append(tampered)
            Y.append(0)
            Y.append(1)
            # report which file has been processed.
            print("Finished processing file: {}".format(file))
            # track how many files were processed.
            count += 1

    # save data as .npz in save_path
    # when loading this .npz file, use array['data_X'], array['data_Y'] for access
    np.savez_compressed(save_path + "\\data.npz", data_X=np.array(X), data_Y=np.array(Y))

if __name__ == "__main__":
    main()