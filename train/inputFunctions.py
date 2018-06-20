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


def getImgAndCommandList(recordingsFolder, printInfo=False,
                         onlyUseSubfolder=None, roundNdigits=3,
                         framesTimeTrashhold=None, cmdTrashhold=0.01,
                         filterZeros=False,
                         getFullCmdList=False):
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

    # If image was token after last cmd, than return last cmd, else...
    if thisTimestamp > cmdList[-1]["timestamp"]:
        if printInfo: print("image was token after last cmd")
        sumValX = float(cmdList[-1]["velX"])
        sumValYaw = float(cmdList[-1]["velYaw"])
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
                 multi_dir=None, augment=0, encode_time=False):
        """Initialization"""
        self.dir = dir
        self.dim = dim
        self.labeled = labeled
        self.batch_size = batch_size
        self.preprocess_input_fn = preprocess_input
        self.preprocess_target_fn = preprocess_target
        self.crop = crop
        self.image_type = image_type
        self.augment = augment
        self._img_ctrl_pairs = []

        assert take_or_skip != 0 and not encode_time

        if multi_dir is None:
            self.img_ctrl_pairs = for_subfolders_in(dir, load_img_ctrl_pairs, as_dict=False)
        else:
            for sub_dir in multi_dir:
                self.img_ctrl_pairs += for_subfolders_in(os.path.join(dir, sub_dir), load_img_ctrl_pairs, as_dict=False)

        if len(self.img_ctrl_pairs) == 0:
            tf.logging.warning("No images found in {}!".format(dir))
        else:
            if take_or_skip > 0:
                self._img_ctrl_pairs = self._img_ctrl_pairs[::take_or_skip]
            elif take_or_skip < 0:
                self._img_ctrl_pairs = [p for i, p in enumerate(self._img_ctrl_pairs) if i % abs(take_or_skip) != 0]

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self._img_ctrl_pairs))

        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self._img_ctrl_pairs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Select image paths and labels with these indexes and load them
        x_batch = self.__data_generation([self._img_ctrl_pairs[k] for k in indexes])

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
        return self._img_ctrl_pairs["angular_z"]

    @property
    def features(self):
        return self._img_ctrl_pairs["img_paths"]

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


def first(arr):
    assert len(arr) > 0

    return arr[0]


def pick_between(ctrl_data, timestamp_start_ns, timestamp_end_ns, reduce_fn=np.mean, max_delay=6e7):
    diff = timestamp_end_ns - timestamp_start_ns
    ctrl_timestamps_ns = np.asarray(ctrl_data[:, 0] * 1e9, dtype=int)
    ctrl_timestamps_rel_start = ctrl_timestamps_ns - timestamp_start_ns
    ctrl_timestamps_rel_end = ctrl_timestamps_ns - timestamp_end_ns

    mask_rel_start = np.greater_equal(ctrl_timestamps_rel_start, 0)
    mask_rel_end = np.less_equal(ctrl_timestamps_rel_end, 0)
    mask = mask_rel_start & mask_rel_end

    if np.sum(mask) == 0 or diff > max_delay:
        linear_x = None
        angular_z = None
    else:
        linear_x = reduce_fn(ctrl_data[mask, 1])
        angular_z = reduce_fn(ctrl_data[mask, 6])

    return [linear_x, angular_z, linear_x != 0.0 or angular_z != 0.0, linear_x is not None and angular_z is not None]


def load_img_ctrl_pairs(data_dir, subdir, image_dir="left_rect", reduce_fn=np.mean, file_type="jpg", filter_zeros=True):
    # Load all images in the subdirectory
    img_paths = np.asarray(glob.glob(os.path.join(data_dir, subdir, image_dir, "*.{}".format(file_type))))

    # Convert img file name to integer timestamp, sort them and calculate start_time as well as timestamp differences
    img_timestamps_ns = np.asarray(sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in img_paths]))

    # Load control data from csv
    ctrl_data = np.loadtxt(os.path.join(data_dir, "cmd_vel_{}.csv".format(subdir)), delimiter=",", dtype=float)
    ctrl_start_ns = int(ctrl_data[0, 0] * 1e9)
    ctrl_end_ns = int(ctrl_data[-1, 0] * 1e9)

    # Pick controls according to the reduce function
    picked_ctrls = []

    for i in range(len(img_timestamps_ns) - 1):
        current_ctrl = pick_between(ctrl_data, img_timestamps_ns[i], img_timestamps_ns[i + 1], reduce_fn=reduce_fn)

        # In case fill missing is active and a ctrl value is missing
        if not bool(current_ctrl[3]):
            if len(picked_ctrls) > 0 and ctrl_start_ns <= img_timestamps_ns[i] <= ctrl_end_ns:
                current_ctrl = picked_ctrls[-1]
            else:
                current_ctrl = [0.0, 0.0, float(not filter_zeros), 0.0]

        picked_ctrls.append(current_ctrl)

    picked_ctrls = np.asarray(picked_ctrls)
    img_paths = img_paths[:-1]

    if filter_zeros:
        mask = np.asarray(picked_ctrls[:, 2], dtype=bool)
        picked_ctrls = picked_ctrls[mask, :]
        img_paths = img_paths[mask]
        img_timestamps_ns = img_timestamps_ns[:-1]
        img_timestamps_ns = img_timestamps_ns[mask]

    start_time_ns = img_timestamps_ns[0]
    rel_ns = np.asarray([(t - start_time_ns) for t in img_timestamps_ns])
    diff_ns = rel_ns[1:] - rel_ns[:-1]

    return {"img_paths": img_paths, "linear_x": picked_ctrls[:, 0], "angular_z": picked_ctrls[:, 1],
            "timestamps": img_timestamps_ns, "time_diff": diff_ns, "time_rel": rel_ns}


def for_subfolders_in(data_dir, apply_fn, as_dict=True):
    subdirs = [d.replace(data_dir, "").replace(os.path.sep, "") for d in glob.glob(os.path.join(data_dir, "*", ""))]
    result = {} if as_dict else []

    for subdir in subdirs:
        tmp_result = apply_fn(data_dir, subdir)

        if as_dict:
            result[subdir] = tmp_result
        else:
            result.append(tmp_result)

    if not as_dict:
        img_paths = np.concatenate([r['img_paths'] for r in result])
        linear_x = np.concatenate([r['linear_x'] for r in result])
        angular_z = np.concatenate([r['angular_z'] for r in result])

        return np.array([(p, x, z) for p, x, z in zip(img_paths, linear_x, angular_z)], dtype=[('img_paths', '<U256'),
                                                                                               ('linear_x', np.float32),
                                                                                               ('angular_z', np.float32)])
    else:
        return result


if __name__ == "__main__":
    robert_ctrl_list = getImgAndCommandList("/home/florian/Development/tmp/data/train_lane",
                                            onlyUseSubfolder="left_rect",
                                            framesTimeTrashhold=None,
                                            filterZeros=False,
                                            printInfo=False)
    robert_ctrl_list_filtered = [s for s in robert_ctrl_list if "straight_lane_fw_4" in s["folderPath"]]
    robert_timestamps_s = np.asarray([int(s["fileName"]) * 1e-9 for s in robert_ctrl_list_filtered])
    robert_angular_z = np.asarray([s["velYaw"] for s in robert_ctrl_list_filtered])
    robert_linear_x = np.asarray([s["velX"] for s in robert_ctrl_list_filtered])

    ####
    data_dir = "/home/florian/Development/tmp/data/train_lane"
    folder = "straight_lane_fw_4"
    dirs = glob.glob(os.path.join(data_dir, "*", ""))
    image_dir = "left_rect"

    files = glob.glob(os.path.join(data_dir, folder, image_dir, "*.jpg"))

    img_timestamps_ns = np.asarray(sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in files]))
    start_time_ns = img_timestamps_ns[0]
    rel_ms = np.asarray([(t - start_time_ns) * 1e-6 for t in img_timestamps_ns])
    diff_ms = rel_ms[1:] - rel_ms[:-1]

    import matplotlib.pyplot as plt
    # plt.hist(diff_ms, bins=21)
    # plt.show()

    ctrl_data = np.loadtxt("/home/florian/Development/tmp/data/train_lane/cmd_vel_{}.csv".format(folder),
                           delimiter=",", dtype=float)


    data_timestamps_s = (ctrl_data[:, 0] - (start_time_ns * 1e-9))
    robert_timestamps_s -= start_time_ns * 1e-9

    # GROUND TRUTH
    plt.plot(data_timestamps_s, ctrl_data[:, 6], label="$angular_z$", linewidth=2)
    plt.plot(data_timestamps_s, ctrl_data[:, 1], label="$linear_x$", linewidth=2)
    #plt.plot(data_timestamps_s, ctrl_data[:, 1] * (-4 * ctrl_data[:, 6]), label="$normed_z$", linewidth=2)

    # ROBERT
    plt.plot(robert_timestamps_s, robert_angular_z, label="Robert: $angular_z$", linewidth=3, linestyle="--")
    plt.plot(robert_timestamps_s, robert_linear_x, label="Robert: $angular_x$", linewidth=3, linestyle="--")

    # MEAN OR FIRST
    pairs = load_img_ctrl_pairs(data_dir, folder, reduce_fn=np.mean)
    plt.scatter((pairs["timestamps"] - start_time_ns) * 1e-9, pairs["angular_z"], label="Nearest: $angular_z$", marker="x")
    plt.scatter((pairs["timestamps"] - start_time_ns) * 1e-9, pairs["linear_x"], label="Nearest: $linear_x$", marker="o")

    #plt.xlim([6.8, 7.8])
    plt.legend()
    plt.grid()
    plt.show()

    exit(0)
    img_ctrl_grouped = for_subfolders_in(data_dir, load_img_ctrl_pairs)

    t = None

    for folder, sample_list in img_ctrl_grouped.items():
        for i, s in enumerate(sample_list):
            timestep = int(os.path.splitext(os.path.basename(s["img_paths"]))[0])
            print(timestep)
            exit(0)

            if t is None:
                t = timestep

            if timestep - t < 1000 * 1e6:
                print(i)
                print(timestep - t)
            else:
                print(i)
                print(timestep - t)
                exit(0)

    # recordingsFolder = os.path.join(os.path.expanduser("~"),
    #                                "volksbot/data/train_lane")
    # predictionsJsonPath = os.path.join(os.path.expanduser("~"),
    #                                   "volksbot", "predictions.json")
    # imgAndCmdList = getImgAndCommandList(recordingsFolder,
    #                                     onlyUseSubfolder="left_rect",
    #                                     framesTimeTrashhold=None,
    #                                     filterZeros=False,
    #                                     printInfo=True)
    # imgAndCommandList = addPredictionsToImgAndCommandList(imgAndCmdList,
    #                                                      predictionsJsonPath,
    #                                                      printInfo=True)

    # subfolderList = getSubfolderListOfImgAndCommandList(imgAndCmdList)
