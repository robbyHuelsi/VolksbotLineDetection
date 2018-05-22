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


def getImgAndCommandList(recordingsFolder, printInfo=False, filter=None):
    print("Collecting from: {} with filter {}".format(recordingsFolder, filter))
    cmdVelFiles = []
    imgsFolders = {}
    inputList = []

    for directory, dirnames, filenames in os.walk(recordingsFolder):
        if directory == recordingsFolder:  # Nur erster Durchlauf/Ebene
            for f in filenames:
                fName, fExtension = os.path.splitext(f)
                if fExtension == ".csv":
                    cmdVelFiles.append(fName)
        else:  # Alle unersen Ebenen
            if filenames:  # Nur wenn Dateien im Ordner
                ibf = os.path.basename(os.path.normpath(directory))  # ibf=Imgage Base Folder
                # print(ibf)
                if not filter or filter == ibf:
                    imgsFolders[directory] = filenames

    imgsFolders = collections.OrderedDict(sorted(imgsFolders.items()))

    for imgFolder, imgFiles in imgsFolders.items():
        imgFiles = sorted(imgFiles)

        ibf = os.path.split(os.path.relpath(imgFolder, recordingsFolder))[0]
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
                        lastVelX = inputList[-1]["velX"] if inputList else 0.0
                        lastVelYaw = inputList[-1]["velYaw"] if inputList else 0.0
                        velX, velYaw = meanCmd(cmdDir,
                                               thisFileName,
                                               nextFileName,
                                               lastVelX,
                                               lastVelYaw,
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


def meanCmd(cmdDir, thisImgName, nextImgName, lastVelX, lastVelYaw, printInfo = False):
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

    if countCmds > 0:
        avVelX = sumValX / countCmds
        avVelYaw = sumValYaw / countCmds
    else:
        avVelX = lastVelX if lastVelX else 0.0
        avVelYaw = lastVelYaw if lastVelYaw else 0.0

    if printInfo:
        print("Between ", str(startTimestamp), " and ", str(endTimestamp), "is 1 command:" if countCmds == 1 else " are " + str(countCmds) + " commands:")
        print("av. velX:    ", str(avVelX))
        print("av. velYaw:  ", str(avVelYaw))

    return avVelX, avVelYaw


class ImageBatchGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset_dir, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True,
                 start_ind=None, end_ind=None, preprocess_input_fn=None, img_filter="left_rect"):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.preprocess_input_fn=preprocess_input_fn

        # Create the data list from dataset directory
        data_list = getImgAndCommandList(dataset_dir, filter=img_filter)
        assert data_list is not None, "No images and velocity commands where found!"

        self.labels = [sample["velYaw"] for sample in data_list
                       if (sample["velYaw"], sample["velX"]) != (0.0, 0.0)]
        self.labels = self.labels[start_ind:end_ind]
        self.img_paths = [sample["imgPath"] for sample in data_list
                          if (sample["velYaw"], sample["velX"]) != (0.0, 0.0)]
        self.img_paths = self.img_paths[start_ind:end_ind]

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

    def __std_preprocess_input(self, path):
        # Open, crop, resize and rescale the image
        img = Image.open(path)
        img = img.crop((380, 0, 1100, 720))
        img = img.resize(self.dim, resample=Image.BILINEAR)
        nd_img = (np.float32(img) / 127.5) - 1.0

        return nd_img

    def __data_generation(self, img_paths_batch):
        """ Generates data containing batch_size samples """
        # Initialization
        x_batch = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, img_path in enumerate(img_paths_batch):
            if self.preprocess_input_fn:
                x_batch[i, ] = self.preprocess_input_fn(img_path)
            else:
                x_batch[i, ] = self.__std_preprocess_input(img_path)

        return x_batch


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"), "recordings")
    res = getImgAndCommandList(recordingsFolder, printInfo=True, filter="left_rect")
