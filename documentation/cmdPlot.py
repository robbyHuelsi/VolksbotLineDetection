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
    end = 126
    
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
    fig = plt.figure(figsize=(4.5, 3.5)) 
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
    ax.set_xlim(0, 1.001)
    ax.set_ylim(0, 0.5)
    
    # y axis ticks
    vals = ax.get_yticks()
    vals = [str(int(x*100)) for x in vals]
    ax.set_yticklabels(vals)

    
    # axis labels
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Giergeschwindigkeit [%]')
    
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
    
    ax.axvline(x=imgTimeList[0], color="#8a8b8a", linestyle=":", linewidth=1, label="Aufzeichnungszeitpunkte Bilder")
    for t in imgTimeList[1:]:
        ax.axvline(x=t, color="#8a8b8a", linestyle="--", linewidth=1)
        
    ax.step(imgTimeList, velYawList, where="post", markevery=2, marker="o", markersize=4, color="orange", label="Gemittelte Steuerbefehle")
    ax.bar(cmdFullTimeList, velYawFullCmdList, color="#85be48", width=0.005, label="Aufgezeichnete Steuerbefehle")
    
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.3,
                     box.width, box.height * 0.7])
    
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=1)

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