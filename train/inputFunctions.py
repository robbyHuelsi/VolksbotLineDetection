#  TODO Think about how the input_fn loads images and controls lazy and
#  controlled way.
#  Number of epochs is defined, number of iterations depends
#  on the batch size and number of samples in the training set.

import os
import collections
import csv
from StdSuites.AppleScript_Suite import string


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
                            print inputDir
                            print
        else:
            print "!!! NOT FOUND: ", str(csvFilePath)
            
    if not inputList:
        return inputList
    else:
        print "!!! NO INPUT"

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
    for timestamp, cmd in cmdDir.iteritems():
        if timestamp >= startTimestamp and timestamp < endTimestamp:
            countCmds += 1
            sumValX += float(cmd["valX"])
            sumValYaw += float(cmd["valYaw"])
            
    avVelX = sumValX / countCmds if countCmds > 0 else 0
    avVelYaw = sumValYaw / countCmds if countCmds > 0 else 0
    
    if printInfo:
        print "Between ", str(startTimestamp), " and ", str(endTimestamp), "is 1 command:" if countCmds == 1 else " are " + str(countCmds) + " commands:"
        print "av. velX:    ", str(avVelX)
        print "av. velYaw:  ", str(avVelYaw)

    return avVelX, avVelYaw


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"), "recordings")
    getImgAndCommandList(recordingsFolder, printInfo = True)
