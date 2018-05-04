#  TODO Think about how the input_fn loads images and controls lazy and
#  controlled way.
#  Number of epochs is defined, number of iterations depends
#  on the batch size and number of samples in the training set.

import os
import collections
import csv


def getImgAndCommandList(recordingsFolder):
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
            # print directory
            # tmpImgs = []
            # for f in filenames:
            imgsFolders[directory] = filenames

    imgsFolders = collections.OrderedDict(sorted(imgsFolders.items()))

    for imgFolder, imgFiles in imgsFolders.iteritems():
        imgFiles = sorted(imgFiles)

        ibf = os.path.basename(os.path.normpath(imgFolder))  # ibf=ImgBaseFolde
        csvFilePath = os.path.join(recordingsFolder, "cmd_vel_" + ibf + ".csv")
        if os.path.isfile(csvFilePath):
            cmdDir = getCmdDir(csvFilePath)
            countImgFiles = len(imgFiles)
            for i in range(countImgFiles-1):  # ...until last but one
                thisFileName, thisFileExt = os.path.splitext(imgFiles[i])
                if thisFileExt == ".jpg":
                    # print str(i), ": ", thisFileName
                    nextFileName = ""
                    if i+1 <= countImgFiles:
                        for j in range(i+1, countImgFiles):
                            nextFN, nextFE = os.path.splitext(imgFiles[j])
                            if nextFE == ".jpg":
                                nextFileName = nextFN
                                # print str(j), ": ", nextFileName
                                break
                    if nextFileName != "":
                        inputDir = {}
                        velX, velYaw = meanCmd(cmdDir,
                                               thisFileName,
                                               nextFileName)
                        inputDir["imgPath"] = os.path.join(imgFolder,
                                                           thisFileName
                                                           + thisFileExt)
                        inputDir["velX"] = velX
                        inputDir["velYaw"] = velYaw
        else:
            print "!!! NOT FOUND: ", str(csvFilePath)

        # print imgFolder, ": ", str(imgFiles)


def getCmdDir(path):
    reader = csv.reader(open(path, 'r'))
    cmdDir = {}
    cmd = {}
    for row in reader:
        cmd["valX"] = row[1]
        cmd["valYaw"] = row[6]
        timestamp = float(row[0])
        cmdDir[timestamp] = cmd
    return cmdDir


def meanCmd(cmdDir, thisImgName, nextImgName):
    print cmdDir
    print thisImgName
    print nextImgName
    print ""

    velX = 0
    velYaw = 0

    return velX, velYaw


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"), "recordings")
    getImgAndCommandList(recordingsFolder)
