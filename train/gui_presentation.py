import matplotlib
matplotlib.use("TkAgg")  # Must to be set before import matplotlib.pyplot or tkinter

import os
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
from datetime import datetime
import time
import threading
import numpy as np

import inputFunctions as ifu
import plotFunctions as pfu

# View and Control
class ImgAndCmdWindow():
    def __init__(self, imgAndCmdList, subfolderList, inverseCmds=True, showInfo=False):
        self.fullImgAndCmdList = imgAndCmdList
        self.filteredImgAndCmdList = None
        self.filteredImgAndCmdListLength = None
        self.applySubfolderFilter()

        self.inverseCmds = inverseCmds
        self.frameNumber = 0
        self.showInfo = showInfo
        self.window = Tk()
        self._job = None  # Necessary for changing scale value by hand
        self.player = None

        self.window.config(bg = 'black')

        # self.inputbox.state("zoomed")
        self.windowWidth = 1280

        style = Style()
        style.theme_use('alt')
        style.configure("trueNeg.Horizontal.TProgressbar",
                        troughcolor='dark green', background="black", thickness=32)
        style.configure("truePos.Horizontal.TProgressbar",
                        troughcolor="black", background='dark green', thickness=33)
        style.configure("true.Label", background="black", foreground='dark green', font=("Helvetica", 30))
        
        
        self.window.title("Autonomous Volksbot")

        self.svSubfolders = StringVar(self.window)
        self.svSubfolders.set("All folders")  # set the default option
        self.omSubfolders = OptionMenu(self.window, self.svSubfolders,
                                       *(["", "All folders"] + [d["folderPath"] for d in subfolderList]),
                                       command=self._omSubfoldersChanged)
        self.omSubfolders.grid(row=0, columnspan=9, sticky="EW")

        self.lImgFrame = Label(self.window)
        self.lImgFrame.grid(row=1, columnspan=9)
        
        self.pbTrueVelYawNeg = Progressbar(self.window, orient=HORIZONTAL,
                                           length=4.0*(self.windowWidth/9)+32, mode="determinate",
                                           style="trueNeg.Horizontal.TProgressbar")
        self.pbTrueVelYawNeg["maximum"] = 1
        self.pbTrueVelYawNeg["value"] = 0
        self.pbTrueVelYawNeg.grid(row=1, column=0, columnspan=3, sticky="S")
        

        self.pbTrueVelYawPos = Progressbar(self.window, orient=HORIZONTAL,
                                           length=4.0*(self.windowWidth/9)+32, mode="determinate",
                                           style="truePos.Horizontal.TProgressbar")
        self.pbTrueVelYawPos["maximum"] = 1
        self.pbTrueVelYawPos["value"] = 0
        self.pbTrueVelYawPos.grid(row=1, column=5, columnspan=3, sticky="S")


        self.svTrueVelYaw = StringVar(value="0 %")
        Label(self.window, textvariable=self.svTrueVelYaw, style="true.Label", width=5, anchor="center").grid(row=1, column=4, sticky="S")
        

        self.scaleFrameNumber = Scale(self.window, from_=0,
                            to=1,
                            length=self.windowWidth, orient=HORIZONTAL,
                            command=self._scaleFrameNumberChanged)
        self.scaleFrameNumber.grid(row=3, columnspan=9)

        bBackward = Button(self.window, text="Backward",
                           width=10, command=self._bBackwardClicked)
        bBackward.grid(row=4, column=0, columnspan=3, sticky="W")

        bForward = Button(self.window, text="Forward",
                          width=10, command=self._bForwardClicked)
        bForward.grid(row=4, column=6, columnspan=3, sticky="E")

        self.bPlayStop = Button(self.window, text="Play",
                                width=10, command=self._bPlayPausedClicked)
        self.bPlayStop.grid(row=4, column=3, columnspan=3)
        
        self.svPath = StringVar(value="/path/to/img")
        self.svDate = StringVar(value="dd.mm.yyyy hh:mm:ss xxxx")
        self.svFrameNumber = StringVar(value="x of i")
        
        bShowPlot = Button(self.window, text="Show Plot", width=10, command=self._bShowPlotClicked)
        bShowPlot.grid(row=5, column=6, columnspan=3, sticky="E")
        
        self.bShowInfo = Button(self.window,
                                text="Hide Info" if self.showInfo else "Show Info",
                                width=10, command=self._bShowInfoClicked)
        self.bShowInfo.grid(row=5, column=3, columnspan=3)

        self.infoFrame = None
        if self.showInfo: self.drawInfoFrame()

        #second_win = Toplevel()
        #self.cmdWindow = ImgAndCmdWindow.CmdWindow(second_win)
        #second_win.update_idletasks()
        #size = tuple(int(_) for _ in second_win.geometry().split('+')[0].split('x'))
        #second_win.geometry("%dx%d+%d+%d" % (size + (3510, 400)))

        # self.window.protocol("WM_DELETE_WINDOW", self.onClosing())
        self.updateViewForSubfolderFilter()
        self.updateViewForFrame()
        
        #self.window.update_idletasks()
        #size = tuple(int(_) for _ in self.window.geometry().split('+')[0].split('x'))
        #self.window.geometry("%dx%d+%d+%d" % (size + (2220, 100)))
        
        self.window.mainloop()
        
    def drawInfoFrame(self):
        self.infoFrame = Frame(self.window)
        Label(self.infoFrame, text="Path: ").grid(sticky="E", row=0, column=0)
        Entry(self.infoFrame, textvariable=self.svPath, state='readonly', width=100).grid(sticky="W", row=0, column=1)
        Label(self.infoFrame, text="Date: ").grid(sticky="E", row=1, column=0)
        Label(self.infoFrame, textvariable=self.svDate).grid(sticky="W", row=1, column=1)
        Label(self.infoFrame, text="Frame: ").grid(sticky="E", row=2, column=0)
        Label(self.infoFrame, textvariable=self.svFrameNumber).grid(sticky="W", row=2, column=1)
        self.infoFrame.grid(row=6, column=0, columnspan=3, sticky="EW")

    def updateViewForSubfolderFilter(self):
        self.scaleFrameNumber.set(0)
        self.scaleFrameNumber.configure(to=self.filteredImgAndCmdListLength)

    def updateViewForFrame(self, frameNumber=-1):
        if frameNumber > -1:
            self.frameNumber = frameNumber

        thisImgAndCmdDict = self.filteredImgAndCmdList[self.frameNumber]
        imgPath = ifu.getImgPathByImgAndCmdDict(thisImgAndCmdDict)
        trueVelX = thisImgAndCmdDict["velX"]
        trueVelX = trueVelX*-1.0 if self.inverseCmds else trueVelX
        trueVelYaw = thisImgAndCmdDict["velYaw"]
        trueVelYaw = trueVelYaw*-1.0 if self.inverseCmds else trueVelYaw
        if "predVelX" in thisImgAndCmdDict:
            predVelX = thisImgAndCmdDict["predVelX"]
            predVelX = predVelX*-1.0 if self.inverseCmds else predVelX
        else:
            predVelX = None
        if "predVelYaw" in thisImgAndCmdDict:
            predVelYaw = thisImgAndCmdDict["predVelYaw"]
            predVelYaw = predVelYaw*-1.0 if self.inverseCmds else predVelYaw
        else:
            predVelYaw = None

        img = Image.open(imgPath)
        img = img.resize((self.windowWidth,
                          int(img.size[1]/img.size[0]*self.windowWidth)),
                         Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(img)
        self.lImgFrame.configure(image=itkFrame)
        self.lImgFrame.image = itkFrame
        
        if self.showInfo:
            self.svPath.set(imgPath)
            self.svDate.set(str(thisImgAndCmdDict["dateTime"]))
            self.svFrameNumber.set(str(self.frameNumber) + " of " + str(self.filteredImgAndCmdListLength))
        
        
        self.scaleFrameNumber.set(self.frameNumber)
        
        #self.cmdWindow.updateViewForFrame(trueVelX, trueVelYaw, predVelX, predVelYaw)
        self.pbTrueVelYawPos["value"] = trueVelYaw if trueVelYaw > 0.0 else 0
        self.pbTrueVelYawNeg["value"] = 1.0 + trueVelYaw if trueVelYaw < 0.0 else 1
        self.svTrueVelYaw.set(str(round(trueVelYaw*100)) + " %")

    def applySubfolderFilter(self, folderPath=None):
        if not folderPath:
            self.filteredImgAndCmdList = self.fullImgAndCmdList
        else:
            self.filteredImgAndCmdList = [e for e in self.fullImgAndCmdList if e["folderPath"] == folderPath]
        self.filteredImgAndCmdListLength = len(self.filteredImgAndCmdList)

    def forward(self):
        if self.frameNumber < self.filteredImgAndCmdListLength-1:
            self.frameNumber += 1
        else:
            self.frameNumber = 0
        self.updateViewForFrame()

    def backward(self):
        if self.frameNumber > 0:
            self.frameNumber -= 1
        else:
            self.frameNumber = self.filteredImgAndCmdListLength-1
        self.updateViewForFrame()

    def _omSubfoldersChanged(self, value):
        if value == "All folders":
            self.applySubfolderFilter()
        else:
            self.applySubfolderFilter(value)
        self.updateViewForSubfolderFilter()
        self.updateViewForFrame(0)

    def _scaleFrameNumberChanged(self, event):
        if self._job:
            self.window.after_cancel(self._job)
        self._job = self.window.after(500, self._updateFrameNumberByScale)

    def _updateFrameNumberByScale(self):
        self._job = None
        self.updateViewForFrame(int(self.scaleFrameNumber.get()))

    def _bForwardClicked(self):
        self.forward()

    def _bBackwardClicked(self):
        self.backward()

    def _bPlayPausedClicked(self):
        if not self.player:
            self.player = self.Player(parent=self)
            self.player.start()
            self.bPlayStop.config(text="Stop")
        else:
            if self.player.isAlive():
                self.player.stop()
            self.player = None
            self.bPlayStop.config(text="Play")
        
    def _bShowInfoClicked(self):
        if self.showInfo:
            # Hiding
            self.showInfo = False
            self.bShowInfo.config(text="Show Info")
            self.svPath.set("")
            self.svDate.set("")
            self.svFrameNumber.set("")
            self.infoFrame.grid_forget()
            self.infoFrame.pack_forget()
            self.infoFrame.place_forget()
            self.infoFrame = None
        else:
            self.showInfo = True
            self.bShowInfo.config(text="Hide Info")
            self.drawInfoFrame()
            
        self.updateViewForFrame()
        
    def _bShowPlotClicked(self):
        refs = [d["velYaw"] for d in self.filteredImgAndCmdList]
        preds = [p["predVelYaw"] if "predVelYaw" in p else None for p in self.filteredImgAndCmdList]
        pfu.plot_ref_pred_comparison(refs, preds)

    def onClosing(self):
        #if self.player:
        #    self.player.stop()
        pass

    class CmdWindow():
        def __init__(self, master):
            self.master = master
            self.frame = Frame(self.master)
            # self.frame.title("Commands" if not self.master.inverseCmds else "Commands (inverted)")

            bgColor = self.master.cget('bg')
            style = Style()
            style.theme_use('alt')
            style.configure("trueNeg.Horizontal.TProgressbar",
                            troughcolor='dark green', background=bgColor)
            style.configure("truePos.Horizontal.TProgressbar",
                            troughcolor=bgColor, background='dark green')
            style.configure("predNeg.Horizontal.TProgressbar",
                            troughcolor='blue', background=bgColor)
            style.configure("predPos.Horizontal.TProgressbar",
                            troughcolor=bgColor, background='blue')
            style.configure("trueNeg.Vertical.TProgressbar",
                            troughcolor='dark green', background=bgColor)
            style.configure("truePos.Vertical.TProgressbar",
                            troughcolor=bgColor, background='dark green')
            style.configure("predNeg.Vertical.TProgressbar",
                            troughcolor='blue', background=bgColor)
            style.configure("predPos.Vertical.TProgressbar",
                            troughcolor=bgColor, background='blue')
            style.configure("true.Label", foreground='dark green')
            style.configure("pred.Label", foreground='blue')

            self.pbTrueVelXPos = Progressbar(self.frame, orient=VERTICAL,
                                             length=200, mode="determinate",
                                             style="truePos.Vertical.TProgressbar")
            self.pbTrueVelXPos["maximum"] = 1
            self.pbTrueVelXPos["value"] = 0
            self.pbTrueVelXPos.grid(row=1, column=4, rowspan=3)

            self.pbPredVelXPos = Progressbar(self.frame, orient=VERTICAL,
                                             length=200, mode="determinate",
                                             style="predPos.Vertical.TProgressbar")
            self.pbPredVelXPos["maximum"] = 1
            self.pbPredVelXPos["value"] = 0
            self.pbPredVelXPos.grid(row=1, column=5, rowspan=3)

            self.pbTrueVelYawNeg = Progressbar(self.frame, orient=HORIZONTAL,
                                               length=200, mode="determinate",
                                               style="trueNeg.Horizontal.TProgressbar")
            self.pbTrueVelYawNeg["maximum"] = 1
            self.pbTrueVelYawNeg["value"] = 0
            self.pbTrueVelYawNeg.grid(row=4, column=1, columnspan=3)

            self.pbPredVelYawNeg = Progressbar(self.frame, orient=HORIZONTAL,
                                               length=200, mode="determinate",
                                               style="predNeg.Horizontal.TProgressbar")
            self.pbPredVelYawNeg["maximum"] = 1
            self.pbPredVelYawNeg["value"] = 0
            self.pbPredVelYawNeg.grid(row=5, column=1, columnspan=3)

            self.pbTrueVelYawPos = Progressbar(self.frame, orient=HORIZONTAL,
                                               length=200, mode="determinate",
                                               style="truePos.Horizontal.TProgressbar")
            self.pbTrueVelYawPos["maximum"] = 1
            self.pbTrueVelYawPos["value"] = 0
            self.pbTrueVelYawPos.grid(row=4, column=6, columnspan=3)

            self.pbPredVelYawPos = Progressbar(self.frame, orient=HORIZONTAL,
                                               length=200, mode="determinate",
                                               style="predPos.Horizontal.TProgressbar")
            self.pbPredVelYawPos["maximum"] = 1
            self.pbPredVelYawPos["value"] = 0
            self.pbPredVelYawPos.grid(row=5, column=6, columnspan=3)

            self.pbTrueVelXNeg = Progressbar(self.frame, orient=VERTICAL,
                                             length=200, mode="determinate",
                                             style="trueNeg.Vertical.TProgressbar")
            self.pbTrueVelXNeg["maximum"] = 1
            self.pbTrueVelXNeg["value"] = 0
            self.pbTrueVelXNeg.grid(row=6, column=4, rowspan=3)

            self.pbPredVelXNeg = Progressbar(self.frame, orient=VERTICAL,
                                             length=200, mode="determinate",
                                             style="predNeg.Vertical.TProgressbar")
            self.pbPredVelXNeg["maximum"] = 1
            self.pbPredVelXNeg["value"] = 0
            self.pbPredVelXNeg.grid(row=6, column=5, rowspan=3)

            Label(self.frame, text="- v_X").grid(row=0, column=4, columnspan=2)
            Label(self.frame, text="+ v_X").grid(row=9, column=4, columnspan=2)
            Label(self.frame, text="+ v_Yaw").grid(row=4, column=0, rowspan=2)
            Label(self.frame, text="- v_Yaw").grid(row=4, column=9, rowspan=2)

            self.svTrueVelXPos = StringVar(value="TrueVelXPos")
            self.svPredVelXPos = StringVar(value="PredVelXPos")
            self.svTrueVelXNeg = StringVar(value="TrueVelXNeg")
            self.svPredVelXNeg = StringVar(value="PredVelXNeg")
            self.svTrueVelYawPos = StringVar(value="TrueVelYawPos")
            self.svPredVelYawPos = StringVar(value="PredVelYawPos")
            self.svTrueVelYawNeg = StringVar(value="TrueVelYawNeg")
            self.svPredVelYawNeg = StringVar(value="PredVelYawNeg")

            labelWidth = 5
            Label(self.frame, textvariable=self.svTrueVelXPos, style="true.Label", width=labelWidth, anchor="e").grid(row=1, column=3, rowspan=3, sticky="E")
            Label(self.frame, textvariable=self.svPredVelXPos, style="pred.Label", width=labelWidth, anchor="w").grid(row=1, column=6, rowspan=3, sticky="W")
            Label(self.frame, textvariable=self.svTrueVelXNeg, style="true.Label", width=labelWidth, anchor="e").grid(row=6, column=3, rowspan=3, sticky="E")
            Label(self.frame, textvariable=self.svPredVelXNeg, style="pred.Label", width=labelWidth, anchor="w").grid(row=6, column=6, rowspan=3, sticky="W")
            Label(self.frame, textvariable=self.svTrueVelYawPos, style="true.Label", width=labelWidth, anchor="center").grid(row=3, column=6, columnspan=3, sticky="S")
            Label(self.frame, textvariable=self.svPredVelYawPos, style="pred.Label", width=labelWidth, anchor="center").grid(row=6, column=6, columnspan=3, sticky="N")
            Label(self.frame, textvariable=self.svTrueVelYawNeg, style="true.Label", width=labelWidth, anchor="center").grid(row=3, column=1, columnspan=3, sticky="S")
            Label(self.frame, textvariable=self.svPredVelYawNeg, style="pred.Label", width=labelWidth, anchor="center").grid(row=6, column=1, columnspan=3, sticky="N")

            Label(self.frame, text="green = true data", style="true.Label").grid(row=10, column=0, columnspan=5)
            Label(self.frame, text="blue = predicted data", style="pred.Label").grid(row=10, column=6, columnspan=5)
            
            self.frame.pack()
            

        def updateViewForFrame(self, trueVelX, trueVelYaw, predVelX, predVelYaw):
            self.pbTrueVelXPos["value"] = trueVelX if trueVelX > 0.0 else 0
            self.pbTrueVelXNeg["value"] = 1.0 + trueVelX if trueVelX < 0.0 else 1
            self.pbPredVelXPos["value"] = predVelX if predVelX and predVelX > 0.0 else 0
            self.pbPredVelXNeg["value"] = 1.0 + predVelX if predVelX and predVelX < 0.0 else 1
            self.pbTrueVelYawPos["value"] = trueVelYaw if trueVelYaw > 0.0 else 0
            self.pbTrueVelYawNeg["value"] = 1.0 + trueVelYaw if trueVelYaw < 0.0 else 1
            self.pbPredVelYawPos["value"] = predVelYaw if predVelYaw and predVelYaw > 0.0 else 0
            self.pbPredVelYawNeg["value"] = 1.0 + predVelYaw if predVelYaw and predVelYaw < 0.0 else 1

            trueVelXPosText = str(round(trueVelX*100)) + " %" if trueVelX > 0 else ""
            trueVelXNegText = str(round(trueVelX*100)) + " %" if trueVelX < 0 else ""
            trueVelYawPosText = str(round(trueVelYaw*100)) + " %" if trueVelYaw > 0 else ""
            trueVelYawNegText = str(round(trueVelYaw*100)) + " %" if trueVelYaw <  0 else ""
            if predVelX:
                predVelXPosText = str(round(predVelX*100)) + " %" if predVelX > 0 else ""
                predVelXNegText = str(round(predVelX*100)) + " %" if predVelX < 0 else ""
            else:
                predVelXPosText = "---"
                predVelXNegText = "---"
            if predVelYaw:
                predVelYawPosText = str(round(predVelYaw*100)) + " %" if predVelYaw > 0 else ""
                predVelYawNegText = str(round(predVelYaw*100)) + " %" if predVelYaw < 0 else ""
            else:
                predVelYawPosText = "---"
                predVelYawNegText = "---"

            self.svTrueVelXPos.set(trueVelXPosText)
            self.svPredVelXPos.set(predVelXPosText)
            self.svTrueVelXNeg.set(trueVelXNegText)
            self.svPredVelXNeg.set(predVelXNegText)
            self.svTrueVelYawPos.set(trueVelYawPosText)
            self.svPredVelYawPos.set(predVelYawPosText)
            self.svTrueVelYawNeg.set(trueVelYawNegText)
            self.svPredVelYawNeg.set(predVelYawNegText)

            # print(trueVelX)
            # print(self.pbTrueVelXNeg["value"])
            # print(self.pbTrueVelYawNeg["value"])

    class Player(threading.Thread):
        def __init__(self, parent=None):
            self.parent = parent
            super(ImgAndCmdWindow.Player, self).__init__()
            self._stop = threading.Event()

        def run(self):
            while not self.stopped():
                self.parent.forward()
                time.sleep(0.04)
 
        def stop(self):
            self._stop.set()

        def stopped(self):
            return self._stop.isSet()


'''
    def getInfo(self):
        #self.inputbox.wait_window()
        return self.status, self.labelId, self.VelX, self.VelYaw
'''


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"),
                                    "volksbot", "data", "aufnahmen_2018_05_22")
    #predictionsJsonPath = os.path.join(os.path.expanduser("~"),
    #                                   "volksbot", "test_course_predictions",
    #                                   "mobilenet_cls_aug.json")

    imgAndCmdList = ifu.getImgAndCommandList(recordingsFolder,
                                             onlyUseSubfolder="left_rect",
                                             filterZeros=False,
                                             useDiscretCmds=False)
    #imgAndCmdList = ifu.addPredictionsToImgAndCommandList(imgAndCmdList,
    #                                                      predictionsJsonPath,
    #                                                      roundNdigits=0)
    subfoldersList = ifu.getSubfolderListOfImgAndCommandList(imgAndCmdList)
    app = ImgAndCmdWindow(imgAndCmdList, subfoldersList, showInfo=False)
