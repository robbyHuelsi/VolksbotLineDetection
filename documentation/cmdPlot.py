import os
import sys
import matplotlib.pyplot as plt

sys.path.append("../train")
import inputFunctions as ifu


def plotCmd(imgAndCmdList):
    start = 100
    end = 151
    startTime = getTimestampByImgAndCmdDict(imgAndCmdList[start])
    
    imgTimeList = []
    cmdFullTimeList = []
    yawVelFullCmdList = []
    yawVelList = []
    
    for imgAndCmdDict in imgAndCmdList[start:end]:
        imgTime = getTimestampByImgAndCmdDict(imgAndCmdDict) - startTime
        imgTimeList.append(imgTime)
        yawVelList.append(float(imgAndCmdDict["velYaw"]))
        cmdDir = imgAndCmdDict["fullCmdList"]
        for cmdTimestamp, cmd in cmdDir.items():
            cmdTime = cmdTimestamp - startTime
            cmdFullTimeList.append(cmdTime)
            yawVelFullCmdList.append(float(cmd["valYaw"]))
        print(imgTime)
        print(imgAndCmdDict)
            
    ax = plt.subplot(111)
    ax.plot(imgTimeList, yawVelList, marker="o", linestyle="none")
    ax.bar(cmdFullTimeList, yawVelFullCmdList, width=0.005)
    for t in imgTimeList:
        ax.axvline(x=t, color="grey")
    
    # set the y-spine (see below for more info on `set_position`)
    ax.spines['bottom'].set_position('zero')
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # limit view
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 0.5)

    
    plt.show()
        
def getTimestampByImgAndCmdDict(imgAndCmdDict):
    return float(imgAndCmdDict["fileName"][:10] + "." + imgAndCmdDict["fileName"][11:])
        
if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"),
                                    "volksbot", "data", "train_lane")
    onlyUseSubfolder = os.path.join("straight_lane_angle_move_right_1",
                                    "left_rect")
    imgAndCmdList = ifu.getImgAndCommandList(recordingsFolder,
                                             onlyUseSubfolder=onlyUseSubfolder,
                                             filterZeros=False, getFullCmdList=True)
    plotCmd(imgAndCmdList)