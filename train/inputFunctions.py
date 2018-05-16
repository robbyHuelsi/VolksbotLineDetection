#  TODO Think about how the input_fn loads images and controls lazy and
#  controlled way.
#  Number of epochs is defined, number of iterations depends
#  on the batch size and number of samples in the training set.

from __future__ import print_function
import os
import collections
import csv
import numpy as np
import tensorflow as tf
from PIL import Image


def getImgAndCommandList(recordingsFolder, printInfo = False):
    cmdVelFiles = []
    imgsFolders = {}
    inputList = []

    for directory, dirnames, filenames in os.walk(recordingsFolder):
        if directory == recordingsFolder:
            for f in filenames:
                fName, fExtension = os.path.splitext(f)
                if fExtension == ".csv":
                    cmdVelFiles.append(fName)
        else:
            imgsFolders[directory] = filenames

    imgsFolders = collections.OrderedDict(sorted(imgsFolders.items()))

    for imgFolder, imgFiles in imgsFolders.items():
        imgFiles = sorted(imgFiles)

        ibf = os.path.basename(os.path.normpath(imgFolder))  # ibf=ImgBaseFolde
        csvFilePath = os.path.join(recordingsFolder, "cmd_vel_" + ibf + ".csv")
        if os.path.isfile(csvFilePath):
            cmdDir = getCmdDir(csvFilePath)
            countImgFiles = len(imgFiles)
            for i in range(countImgFiles-1):  # ...until last but one
                thisFileName, thisFileExt = os.path.splitext(imgFiles[i])
                if thisFileExt == ".jpg":
                    nextFileName = ""
                    if i+1 <= countImgFiles:
                        for j in range(i+1, countImgFiles):
                            nextFN, nextFE = os.path.splitext(imgFiles[j])
                            if nextFE == ".jpg":
                                nextFileName = nextFN
                                break
                    if nextFileName != "":
                        inputDir = {}
                        velX, velYaw = meanCmd(cmdDir,
                                               thisFileName,
                                               nextFileName,
                                               printInfo)
                        inputDir["imgPath"] = os.path.join(imgFolder,
                                                           thisFileName
                                                           + thisFileExt)
                        inputDir["velX"] = velX
                        inputDir["velYaw"] = velYaw
                        inputList.append(inputDir)
                        
                        if printInfo:
                            print(inputDir)
        else:
            print("!!! NOT FOUND: ", str(csvFilePath))

    if len(inputList) > 0:
        return inputList
    else:
        print("!!! NO INPUT")


def getCmdDir(path):
    reader = csv.reader(open(path, 'r'))
    cmdDir = {}
    for row in reader:
        cmd = {}
        cmd["valX"] = str(row[1])
        cmd["valYaw"] = str(row[6])
        timestamp = float(row[0])
        cmdDir[timestamp] = cmd
    return cmdDir


def meanCmd(cmdDir, thisImgName, nextImgName, printInfo = False):
    startTimestamp = float(thisImgName)/1000000000
    endTimestamp = float(nextImgName)/1000000000
    sumValX = 0
    sumValYaw = 0
    countCmds = 0
    for timestamp, cmd in cmdDir.items():
        if timestamp >= startTimestamp and timestamp < endTimestamp:
            countCmds += 1
            sumValX += float(cmd["valX"])
            sumValYaw += float(cmd["valYaw"])
            
    avVelX = sumValX / countCmds if countCmds > 0 else 0
    avVelYaw = sumValYaw / countCmds if countCmds > 0 else 0
    
    if printInfo:
        print("Between ", str(startTimestamp), " and ", str(endTimestamp), "is 1 command:" if countCmds == 1 else " are " + str(countCmds) + " commands:")
        print("av. velX:    ", str(avVelX))
        print("av. velYaw:  ", str(avVelYaw))

    return avVelX, avVelYaw


class ImageBatchGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset_dir, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True,
                 start_ind=None, end_ind=None):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size

        # Create the data list from dataset directory
        data_list = getImgAndCommandList(dataset_dir)

        self.img_paths = [sample["imgPath"] for sample in data_list]
        self.img_paths = self.img_paths[start_ind:end_ind]
        self.labels = [sample["velYaw"] for sample in data_list]
        self.labels = self.labels[start_ind:end_ind]

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.img_paths))

        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Select image paths and labels with these indexes
        img_paths_batch = [self.img_paths[k] for k in indexes]
        y_batch = [self.labels[k] for k in indexes]

        # Loading the images via path batch
        x_batch = self.__data_generation(img_paths_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_batch):
        """ Generates data containing batch_size samples """
        # Initialization
        x_batch = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, img_path in enumerate(img_paths_batch):
            # Open, crop, resize and rescale the image
            # TODO Make the preprocessing step inside the ImageBatchGenerator more configurable
            img = Image.open(img_path)
            img = img.crop((380, 0, 1100, 720))
            img = img.resize(self.dim, resample=Image.BILINEAR)
            # Tensorflow (especially mobile net) requires pixel values to be in range [-1.0, 1.0]
            nd_img = (np.float32(img) / 127.5) - 1.0
            x_batch[i, ] = nd_img

        return x_batch


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"), "recordings")
    res = getImgAndCommandList(recordingsFolder, printInfo=True)
