import os
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from builtins import str
from _ast import Str

sys.path.append("../train")
import inputFunctions as ifu


def plotCmd(imgAndCmdList):
    start = 100
    end = 125
    
    # sort by dateTime
    #imgAndCmdList = sorted(imgAndCmdList, key=lambda k: k['dateTime'])
    #for imgAndCmdDict in imgAndCmdList:
    #    imgAndCmdDict["fullCmdList"] = sorted(imgAndCmdDict["fullCmdList"],
    #                                          key=lambda k: k['dateTime'])
    
    # get delay
    startTime = imgAndCmdList[start]['dateTime']
    
    # add delay to list
    for imgAndCmdDict in imgAndCmdList:
        imgAndCmdDict["delay"] = (imgAndCmdDict["dateTime"] - startTime).total_seconds()
        #print("IMG: " + str(imgAndCmdDict["delay"]))
        for fullCmdDict in imgAndCmdDict["fullCmdList"]:
            fullCmdDict["delay"] = (fullCmdDict["dateTime"] - startTime).total_seconds()
            #print("     cmd: " + str(fullCmdDict["delay"]))
    #input()
    
    imgTimeList = []
    cmdFullTimeList = []
    velYawFullCmdList = []
    velYawList = []

    # plot figure
    fig = plt.figure(figsize=(10, 3)) 
    ax = plt.subplot(111)
    
    # set the y-spine (see below for more info on `set_position`)
    ax.spines['bottom'].set_position('zero')
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # limit view
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    
    # axis labels
    ax.set_xlabel('Zeit [Sekunden]')
    ax.set_xlabel('Giergeschwindigkeit')
    
    for i, imgAndCmdDict in enumerate(imgAndCmdList[start:end]):
        imgTime = imgAndCmdDict["delay"]
        imgTimeList.append(imgTime)
        velYawList.append(float(imgAndCmdDict["velYaw"]))
        #print(str(i) + ".pdf: " + str(imgTime) + " (mean: " + str(imgAndCmdDict["velYaw"]) + ")")
        for fullCmdDict in imgAndCmdDict["fullCmdList"]:
            cmdTime = fullCmdDict["delay"]
            cmdFullTimeList.append(cmdTime)
            velYawFullCmdList.append(float(fullCmdDict["velYaw"]))
            #print("        - " + str(cmdTime) + ": " + str(fullCmdDict["velYaw"]))
    
    ax.axvline(x=imgTimeList[0], color="grey", linestyle=":", linewidth=1, label="Aufkommen neuer Bilder")
    for t in imgTimeList[1:]:
        ax.axvline(x=t, color="grey", linestyle="--", linewidth=1)
        
    ax.step(imgTimeList, velYawList, where="post", markevery=2, marker="o", color="orange", label="Gemittelte Steuerbefehle")
    ax.bar(cmdFullTimeList, velYawFullCmdList, color="green", width=0.005, label="Alle Steuerbefehle")
    ax.legend()

    plt.savefig('cmds.pdf')
        
    
        
if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"),
                                    "volksbot", "data", "train_lane")
    onlyUseSubfolder = os.path.join("straight_lane_angle_move_right_1",
                                    "left_rect")
    imgAndCmdList = ifu.getImgAndCommandList(recordingsFolder,
                                             onlyUseSubfolder=onlyUseSubfolder,
                                             filterZeros=False, getFullCmdList=True)
    plotCmd(imgAndCmdList)