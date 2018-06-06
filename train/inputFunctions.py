from __future__ import print_function
import os
import collections
import csv
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import decimal


def getImgAndCommandList(recordingsFolder, printInfo=False,
                         onlyUseSubfolder=None, roundNdigits=3, trashhold=0.01,
                         filterZeros=False, predictionsFile=None):
    print("Collecting from: {}".format(recordingsFolder))
    print("but only use subfolder: {}".format(onlyUseSubfolder))
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
                if not onlyUseSubfolder or onlyUseSubfolder == ibf:
                    imgsFolders[directory] = filenames

    imgsFolders = collections.OrderedDict(sorted(imgsFolders.items()))

    for imgFolder, imgFiles in imgsFolders.items():
        imgFiles = sorted(imgFiles)

        ibf = os.path.split(os.path.relpath(imgFolder, recordingsFolder))[0]
        csvFilePath = os.path.join(recordingsFolder, "cmd_vel_" + ibf + ".csv")
        if os.path.isfile(csvFilePath):
            cmdDir = getCmdDir(csvFilePath)
            countImgFiles = len(imgFiles)
            for i in range(countImgFiles - 1):  # ...until last but one
                thisFileName, thisFileExt = os.path.splitext(imgFiles[i])
                if thisFileExt == ".jpg":
                    nextFileName = ""
                    if i + 1 <= countImgFiles:
                        for j in range(i + 1, countImgFiles):
                            nextFN, nextFE = os.path.splitext(imgFiles[j])
                            if nextFE == ".jpg":
                                nextFileName = nextFN
                                break
                    if nextFileName != "":
                        inputDir = {}

                        if inputList and inputList[-1]["folderPath"] == imgFolder:
                            lastVelX = inputList[-1]["velX"]
                            lastVelYaw = inputList[-1]["velYaw"]
                        else:
                            lastVelX = 0.0
                            lastVelYaw = 0.0

                        velX, velYaw = calcCmds(cmdDir,
                                                thisFileName,
                                                nextFileName,
                                                lastVelX,
                                                lastVelYaw,
                                                roundNdigits=roundNdigits,
                                                trashhold=trashhold,
                                                printInfo=printInfo)
                        inputDir["folderPath"] = imgFolder
                        inputDir["fileName"] = thisFileName
                        inputDir["fileExt"] = thisFileExt
                        inputDir["velX"] = velX
                        inputDir["velYaw"] = velYaw
                        inputList.append(inputDir)

                        if printInfo:
                            print(inputDir)
        else:
            print("!!! NOT FOUND: ", str(csvFilePath))

    if len(inputList) > 0:
        if filterZeros:
            inputList = [d for d in inputList if
                         (d['velX'] != 0.0 or d['velYaw'] != 0.0)]
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


def calcCmds(cmdDir, thisImgName, nextImgName, lastVelX, lastVelYaw,
             roundNdigits=3, trashhold=0.0, printInfo=False):
    startTimestamp = float(thisImgName) / 1000000000
    endTimestamp = float(nextImgName) / 1000000000
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

    if roundNdigits and roundNdigits >= 0:
        avVelX = round(avVelX, ndigits=roundNdigits)
        avVelYaw = round(avVelYaw, ndigits=roundNdigits)

    if trashhold and trashhold >= 0.0:
        avVelX = avVelX if abs(avVelX) > trashhold else 0.0
        avVelYaw = avVelYaw if abs(avVelYaw) > trashhold else 0.0

    if printInfo:
        print("Between ", str(startTimestamp), " and ", str(endTimestamp),
              "is 1 command:" if countCmds == 1 else " are " + str(countCmds) + " commands:")
        print("av. velX:    ", str(avVelX))
        print("av. velYaw:  ", str(avVelYaw))

    return avVelX, avVelYaw


def getImgPathByImgAndCmdDict(imgAndCmdDict):
    return os.path.join(imgAndCmdDict["folderPath"],
                        imgAndCmdDict["fileName"]
                        + imgAndCmdDict["fileExt"])


def addPredictionsToImgAndCommandList(imgAndCommandList, predictionsJsonPath,
                                      roundNdigits=0, printInfo=False):
    with open(predictionsJsonPath) as f:
        predictedCmdList = json.load(f)

    if not predictedCmdList:
        print("!!! Loading json file for predicted cmd's failed!")
        print(predictionsJsonPath)
    else:
        for imgAndCmdDict in imgAndCommandList:
            filteredPCL = [d for d in predictedCmdList if
                           (d['fileName'] in imgAndCmdDict['fileName'] and
                            d['fileExt'] in imgAndCmdDict['fileExt'])]
            if len(filteredPCL) == 0:
                print("!!! No predicted cmd for " +
                      getImgPathByImgAndCmdDict(imgAndCmdDict))
            elif len(filteredPCL) > 1:
                print("!!! Multiple predicted cmd's for " +
                      getImgPathByImgAndCmdDict(imgAndCmdDict) + ":")
                for d in filteredPCL:
                    print(d)
            else:
                if "predVelX" in filteredPCL[0]:
                    predVelX = filteredPCL[0]["predVelX"]
                    if roundNdigits and roundNdigits >= 0:
                        predVelX = round(predVelX, ndigits=roundNdigits)
                    imgAndCmdDict["predVelX"] = predVelX
                if "predVelYaw" in filteredPCL[0]:
                    predVelYaw = filteredPCL[0]["predVelYaw"]
                    if roundNdigits and roundNdigits >= 0:
                        predVelYaw = round(predVelYaw, ndigits=roundNdigits)
                    imgAndCmdDict["predVelYaw"] = predVelYaw
                if printInfo:
                    print(imgAndCmdDict)

    return imgAndCommandList


class ImageBatchGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dir, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True, sub_dir="left_rect",
                 preprocess_input=None, preprocess_target=None, labeled=True, crop=True, take_or_skip=0):
        """Initialization"""
        self.dir = dir
        self.dim = dim
        self.labeled = labeled
        self.batch_size = batch_size
        self.preprocess_input_fn = preprocess_input
        self.preprocess_target_fn = preprocess_target
        self._labels = []
        self._img_paths = []
        self.crop = crop

        if labeled:
            data_list = getImgAndCommandList(dir, onlyUseSubfolder=sub_dir, filterZeros=True)
        else:
            # If no labels are needed, search for every image in the directory
            data_list = [{"imgPath": p} for p in glob.glob(os.path.join(dir, "*", "*.jpg"))]

        if data_list is None:
            tf.logging.warning("No images found in {}!".format(dir))
        else:
            self._img_paths = [getImgPathByImgAndCmdDict(sample) for sample in data_list]

            if take_or_skip > 0:
                self._img_paths = self._img_paths[::take_or_skip]
            elif take_or_skip < 0:
                self._img_paths = [p for i, p in enumerate(self._img_paths) if i % abs(take_or_skip) != 0]

            if labeled:
                self._labels = [sample["velYaw"] for sample in data_list]

                if take_or_skip > 0:
                    self._labels = self._labels[::take_or_skip]
                elif take_or_skip < 0:
                    self._labels = [p for i, p in enumerate(self._labels) if i % abs(take_or_skip) != 0]

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self._img_paths))

        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self._img_paths) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Select image paths and labels with these indexes and load them
        x_batch = self.__data_generation([self._img_paths[k] for k in indexes])

        if self.labeled:
            y_batch = [self._labels[k] for k in indexes]

            # If the target value also has to be preprocessed then do it now
            if self.preprocess_target_fn is not None:
                y_batch = self.preprocess_target_fn(y_batch)

            return x_batch, y_batch
        else:
            return x_batch

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __std_preprocess_input(self, path):
        # Open, crop, resize and rescale the image
        img = Image.open(path)

        if self.crop:
            img = img.crop((380, 0, 1100, 720))

        img = img.resize(self.dim, resample=Image.BILINEAR)
        nd_img = (np.float32(img) / 127.5) - 1.0

        return nd_img

    def __data_generation(self, img_paths_batch):
        """ Generates data containing batch_size samples """
        # Initialization
        x_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Generate data
        for i, img_path in enumerate(img_paths_batch):
            if self.preprocess_input_fn:
                x_batch[i, ] = self.preprocess_input_fn(img_path, self.crop)
            else:
                x_batch[i, ] = self.__std_preprocess_input(img_path)

        return x_batch

    @property
    def labels(self):
        return self._labels

    @property
    def features(self):
        return self._img_paths


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"),
                                    "recordings_vs")
    predictionsJsonPath = os.path.join(os.path.expanduser("~"),
                                       "volksbot", "predictions.json")
    imgAndCommandList = getImgAndCommandList(recordingsFolder,
                                             onlyUseSubfolder="left_rect",
                                             filterZeros=True)
    imgAndCommandList = addPredictionsToImgAndCommandList(imgAndCommandList,
                                                          predictionsJsonPath,
                                                          printInfo=True)
