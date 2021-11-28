import csv
import os
import sys
import numpy as np

import buildDataSet

from utils.dicom_utils import load_dicom
import dicom_utils

LOAD_PATHS = []

LABEL_PATHS = []

SAVE_PATH = None

def generateLabels(path, labels):
    with open(path, 'r') as labelFile:
        reader = csv.reader(labelFile)
        for i, row in enumerate(reader):
            # skip if instance already has been labeled as tampered.
            if row[1] in labels and labels[row[1]] == 1:
                continue
            # if tampered, label 1 else label 0 for now.
            # label can change if this slice is untampered but other slices of same instance is tampered.
            labels[row[1]] = 1 if (row[0] == 'FB' or row[0] == 'FM') else 0
            
def main():
    labels = {}
    # for every path containing the labels, generate labels.
    for label_path in LABEL_PATHS:
        generateLabels(label_path, labels)

    devtest_X = []
    devtest_Y = []
    # for every file containing the dev/test set images, generate dataset.
    for path in LOAD_PATHS:
        files = os.listdir(path)
        # process every instance.
        for file in files:
            scan = load_dicom(path + '\\' + file)               # load dcm CT scan as ndarray.
            scan = buildDataSet.normalize(scan[0])              # normalize for HU values.
            scan = buildDataSet.resize_volume(scan)             # resize volume for efficiency.
            devtest_X.append(scan)                              # append formatted ndarray scan.
            devtest_Y.append(labels[file])                      # grab label from dictionary and append to list.

    # save data as .npz in save_path.
    # when loading this .npz file, use array['devtest_X'] and array['devtest_Y'] for access.
    np.savez_compressed(SAVE_PATH + "\\devtest.npz", devtest_X=np.array(devtest_X), devtest_Y=np.array(devtest_Y))

if __name__ == "__main__":
    main()