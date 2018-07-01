from __future__ import print_function

import glob
import json
import os
import collections
import csv
import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from PIL import Image
from datetime import datetime
from generateDataset import pillow_augmentations, gaussian_noise, channel_wise_zero_mean

from models import mobilenet_cls


def getImgAndCommandList(recordingsFolder, printInfo=False,
                         onlyUseSubfolder=None, roundNdigits=3,
                         framesTimeTrashhold=None, cmdTrashhold=0.01,
                         filterZeros=False, getFullCmdList=False,
                         useDiscretCmds=False):
    '''
    onlyUseSubfolder:
    - None:                Use all subfolders
    - e.g. "subfolder":    Use only frames in the last subfolder "subfolder"
    - e.g. "sub/folders":  Use only frames in the last two subfolders "sub/folders"
    '''

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
                if not onlyUseSubfolder:
                    imgsFolders[directory] = filenames
                else:
                    dsl = os.path.normpath(directory)  # ibf=directory splitted list
                    dsl = dsl.split(os.sep)
                    fsl = os.path.normpath(onlyUseSubfolder)  # fsl=onlyUseSubfolder (filter) splitted list
                    fsl = fsl.split(os.sep)
                    filteredSubpathList = dsl[-1 * len(fsl):]
                    filteredSubpath = os.path.join(*filteredSubpathList)
                    # print(dsl)
                    # print(fsl)
                    # print(onlyUseSubfolder)
                    # print(filteredSubpath)
                    if onlyUseSubfolder == filteredSubpath:
                        imgsFolders[directory] = filenames
                        # print("added")
                    # input()

    imgsFolders = collections.OrderedDict(sorted(imgsFolders.items()))

    for imgFolder, filesList in imgsFolders.items():

        # check is there a cmd file
        ibf = os.path.split(os.path.relpath(imgFolder, recordingsFolder))[0]
        csvFilePath = os.path.join(recordingsFolder, "cmd_vel_" + ibf + ".csv")
        if os.path.isfile(csvFilePath):

            # convert cmd file in a list of cmds
            cmdsListFromCSV = getCmdList(csvFilePath)

            # get list of images
            imgsList = []
            for f in filesList:
                name, extension = os.path.splitext(f)
                if extension == ".jpg":
                    imgDir = {}
                    imgDir["fileName"] = name
                    imgDir["fileExt"] = extension
                    timestamp = float(name) / 1000000000
                    imgDir["timestamp"] = timestamp
                    imgDir["dateTime"] = datetime.fromtimestamp(timestamp)
                    imgsList.append(imgDir)

            # sort imgsList by timestamp
            imgsList = sorted(imgsList, key=lambda k: k["fileName"])

            # frames time trashhold
            if framesTimeTrashhold and len(imgsList) > 0:
                filteredImgsList = []
                filteredImgsList.append(imgsList[0])
                for imgDir in imgsList[1:]:
                    diff = (imgDir["dateTime"] - filteredImgsList[-1]["dateTime"]).total_seconds()
                    if diff > float(framesTimeTrashhold):
                        filteredImgsList.append(imgDir)
                imgsList = filteredImgsList

            # get comands and fill in inputList
            for i, imgDir in enumerate(imgsList[:-1]):  # ...until last but one
                inputDir = {}
                if inputList and inputList[-1]["folderPath"] == imgFolder:
                    lastVelX = inputList[-1]["velX"]
                    lastVelYaw = inputList[-1]["velYaw"]
                else:
                    lastVelX = 0.0
                    lastVelYaw = 0.0

                velX, velYaw, filteredCmdList = calcCmds(cmdsListFromCSV,
                                                         imgDir["timestamp"],
                                                         imgsList[i + 1]["timestamp"],
                                                         lastVelX,
                                                         lastVelYaw,
                                                         roundNdigits=roundNdigits,
                                                         cmdTrashhold=cmdTrashhold,
                                                         printInfo=printInfo)
                inputDir["folderPath"] = imgFolder
                inputDir["fileName"] = imgDir["fileName"]
                inputDir["fileExt"] = imgDir["fileExt"]
                inputDir["dateTime"] = imgDir["dateTime"]
                inputDir["velX"] = velX
                inputDir["velYaw"] = velYaw
                if getFullCmdList: inputDir["fullCmdList"] = filteredCmdList
                inputList.append(inputDir)

                if printInfo:
                    print(inputDir)
        else:
            print("!!! NOT FOUND: ", str(csvFilePath))

    if len(inputList) > 0:
        if filterZeros:
            inputList = [d for d in inputList if
                         (d['velX'] != 0.0 or d['velYaw'] != 0.0)]
        if useDiscretCmds:
            mobileNetCls = mobilenet_cls.MobileNetCls()
            for inputDict in inputList:
                valXCls = mobilenet_cls.getVelYawClas(inputDict["velX"])
                valYawCls = mobilenet_cls.getVelYawClas(inputDict["velYaw"])
                valXCls = mobilenet_cls.oneHotEncode(valXCls).reshape((1,-1))
                valYawCls = mobilenet_cls.oneHotEncode(valYawCls).reshape((1,-1))
                valXCls = mobileNetCls.postprocess_output(valXCls)[0]
                valYawCls = mobileNetCls.postprocess_output(valYawCls)[0]
                inputDict["velX"] = valXCls
                inputDict["velYaw"] = valYawCls
        return inputList
    else:
        print("!!! NO INPUT")


def getCmdList(path):
    reader = csv.reader(open(path, 'r'))
    cmdList = []
    for row in reader:
        cmdDir = {}
        cmdDir["velX"] = str(row[1])
        cmdDir["velYaw"] = str(row[6])
        cmdDir["timestamp"] = float(row[0])
        dt = datetime.fromtimestamp(cmdDir["timestamp"])
        cmdDir["dateTime"] = dt
        cmdList.append(cmdDir)
    return cmdList


def calcCmds(cmdList, thisTimestamp, nextTimestamp, lastVelX, lastVelYaw,
             roundNdigits=3, cmdTrashhold=0.0, printInfo=False):

    sumValX = 0
    sumValYaw = 0
    countCmds = 0
    filteredCmdList = []
    
    # If image was token after last cmd, than return zeros, else...
    if thisTimestamp > cmdList[-1]["timestamp"]:
        if printInfo: print("!!! image was token after last cmd")
        # sumValX = float(cmdList[-1]["velX"])
        # sumValYaw = float(cmdList[-1]["velYaw"])
        sumValX = 0.0
        sumValYaw = 0.0
        countCmds = 1
        filteredCmdList.append(cmdList[-1])
    else:
        for cmdDir in cmdList:
            if cmdDir["timestamp"] >= thisTimestamp and cmdDir["timestamp"] < nextTimestamp:
                countCmds += 1
                sumValX += float(cmdDir["velX"])
                sumValYaw += float(cmdDir["velYaw"])
                filteredCmdList.append(cmdDir)

    if countCmds > 0:
        avVelX = sumValX / countCmds
        avVelYaw = sumValYaw / countCmds
    else:
        avVelX = lastVelX if lastVelX else 0.0
        avVelYaw = lastVelYaw if lastVelYaw else 0.0

    if roundNdigits and roundNdigits >= 0:
        avVelX = round(avVelX, ndigits=roundNdigits)
        avVelYaw = round(avVelYaw, ndigits=roundNdigits)

    if cmdTrashhold and cmdTrashhold >= 0.0:
        avVelX = avVelX if abs(avVelX) > cmdTrashhold else 0.0
        avVelYaw = avVelYaw if abs(avVelYaw) > cmdTrashhold else 0.0

    if printInfo:
        print("Between ", str(thisTimestamp), " and ", str(nextTimestamp),
              "is 1 command:" if countCmds == 1 else " are " + str(countCmds) + " commands:")
        print("av. velX:    ", str(avVelX))
        print("av. velYaw:  ", str(avVelYaw))

    return avVelX, avVelYaw, filteredCmdList


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
                if printInfo:
                    print("No predicted cmd for " +
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


def getSubfolderListOfImgAndCommandList(imgAndCmdList):
    subfoldersList = []

    for i, imgAndCmdDict in enumerate(imgAndCmdList):
        thisPath = imgAndCmdDict["folderPath"]
        nextPath = imgAndCmdList[i + 1]["folderPath"] if i < len(imgAndCmdList) - 1 else None
        if i == len(imgAndCmdList) - 1 or thisPath != nextPath:
            # Letzes Element oder das naechste Element gehoert schon zum naechsten Subfolder
            subfolderDict = {}
            subfolderDict["folderPath"] = thisPath
            subfolderDict["startI"] = subfoldersList[-1]["stopI"] + 1 if len(subfoldersList) > 0 else 0
            subfolderDict["stopI"] = i
            subfoldersList.append(subfolderDict)
            # print(subfolderDict)
            # print(len(subfoldersList))
    return subfoldersList


class ImageBatchGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, dir, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True, image_type="left_rect",
                 preprocess_input=None, preprocess_target=None, labeled=True, crop=True, take_or_skip=0,
                 multi_dir=None, augment=0):
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
        self.image_type = image_type
        self.augment = augment

        data_list = []

        if multi_dir is None:
            data_list = getImgAndCommandList(dir, onlyUseSubfolder=image_type, filterZeros=True)
        else:
            for sub_dir in multi_dir:
                data_list += getImgAndCommandList(os.path.join(dir, sub_dir), onlyUseSubfolder=image_type,
                                                  filterZeros=True)

        if len(data_list) == 0:
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

        img = img.resize(self.dim, resample=Image.NEAREST)

        if self.augment:
            img = pillow_augmentations(img)

        img = self.preprocess_input_fn(np.float32(img)) if self.preprocess_input_fn else np.float32(img)

        if self.augment:
            img = gaussian_noise(img)

        #img = channel_wise_zero_mean(img)

        return img

    def __data_generation(self, img_paths_batch):
        """ Generates data containing batch_size samples """
        # Initialization
        x_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Generate data
        for i, img_path in enumerate(img_paths_batch):
            x_batch[i, ] = self.__std_preprocess_input(img_path)

        return x_batch

    @property
    def labels(self):
        return self._labels

    @property
    def features(self):
        return self._img_paths

    @staticmethod
    def from_args_and_helper(args, helper, mode):
        assert mode in ["train", "val", "pred"], "Mode {} is not supported!".format(mode)

        if mode == "train":
            return ImageBatchGenerator(args.data_dir, batch_size=args.batch_size, crop=args.crop,
                                       preprocess_input=helper.preprocess_input,
                                       preprocess_target=helper.preprocess_target,
                                       image_type=args.sub_dir, take_or_skip=(-1 * args.take_or_skip),
                                       multi_dir=args.train_dir, augment=args.augment, shuffle=bool(args.shuffle))
        elif mode == "val":
            return ImageBatchGenerator(args.data_dir, batch_size=args.batch_size, crop=args.crop,
                                       preprocess_input=helper.preprocess_input,
                                       preprocess_target=helper.preprocess_target,
                                       image_type=args.sub_dir, take_or_skip=args.take_or_skip,
                                       multi_dir=args.val_dir, shuffle=False)
        elif mode == "pred":
            return ImageBatchGenerator(args.data_dir, batch_size=1, crop=args.crop,
                                       preprocess_input=helper.preprocess_input,
                                       preprocess_target=helper.preprocess_target,
                                       image_type=args.sub_dir, shuffle=False, take_or_skip=0,
                                       multi_dir=args.val_dir)


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"),
                                    "volksbot", "data", "test_course_oldcfg")
    predictionsJsonPath = os.path.join(os.path.expanduser("~"),
                                       "volksbot/run/mobile_9cls_v6/predictions.json")
    imgAndCmdList = getImgAndCommandList(recordingsFolder,
                                         onlyUseSubfolder="left_rect",
                                         framesTimeTrashhold=None,
                                         filterZeros=False,
                                         useDiscretCmds=True,
                                         printInfo=True)
    # imgAndCommandList = addPredictionsToImgAndCommandList(imgAndCmdList,
    #                                                      predictionsJsonPath,
    #                                                      printInfo=True)

    subfolderList = getSubfolderListOfImgAndCommandList(imgAndCmdList)
