import os
import json
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
import time
import threading

import inputFunctions

# View and Controll
class ImgAndCmdWindow():
    def __init__(self, imgAndCommandList):
        self.imgAndCommandList = imgAndCommandList
        self.i = 0
        self.window = Tk()
        self._job = None  # Necessary for changing scale value by hand
        self.player = None

        # self.inputbox.state("zoomed")

        self.windowWidth = 1200

        self.lImgFrame = Label(self.window)
        self.lImgFrame.grid(row=0, columnspan=3)

        self.scaleI = Scale(self.window, from_=0, to=len(self.imgAndCommandList), length=self.windowWidth, orient=HORIZONTAL, command=self.scaleIUpdated)
        self.scaleI.grid(row=1, columnspan=3)

        '''
        Label(self.window, text="Vel. X").grid(row=2, column=0)
        self.eVelX = Entry(self.window)
        self.eVelX.grid(row=2, column=1)
        self.eVelX.focus_set()
        self.pbVelX = Progressbar(self.window, orient=HORIZONTAL,
                                      length=200, mode="determinate")
        self.pbVelX["maximum"] = 2
        self.pbVelX.grid(row=2, column=2)

        Label(self.window, text="Vel. Yaw").grid(row=3, column=0)
        self.eVelYaw = Entry(self.window)
        self.eVelYaw.grid(row=3, column=1,)
        self.pbVelYaw = Progressbar(self.window, orient=HORIZONTAL,
                                      length=200, mode="determinate")
        self.pbVelYaw["maximum"] = 2
        self.pbVelYaw.grid(row=3, column=2)
        '''

        bBackward = Button(self.window, text="Backward",
                           width=10, command=self._bBackwardClicked)
        bBackward.grid(row=3, column=0)

        bForward = Button(self.window, text="Forward",
                          width=10, command=self._bForwardClicked)
        bForward.grid(row=3, column=2)

        self.bPlayStop = Button(self.window, text="Play",
                                width=10, command=self._bPlayPausedClicked)
        self.bPlayStop.grid(row=3, column=1)

        self.cmdWindow = ImgAndCmdWindow.CmdWindow(Toplevel(self.window))

        # self.window.protocol("WM_DELETE_WINDOW", self.onClosing())
        self.updateView()
        self.window.mainloop()

    def updateView(self, i=-1):
        if i > -1:
            self.i = i

        imgPath = self.imgAndCommandList[self.i]["imgPath"]
        trueVelX = self.imgAndCommandList[self.i]["velX"]
        trueVelYaw = self.imgAndCommandList[self.i]["velYaw"]

        self.window.title(str(imgPath))

        img = Image.open(imgPath)
        img = img.resize((self.windowWidth, int(img.size[1]/img.size[0]*self.windowWidth)), Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(img)
        self.lImgFrame.configure(image=itkFrame)
        self.lImgFrame.image = itkFrame

        self.scaleI.set(self.i)

        '''
        self.eVelX.delete(0, END)
        self.eVelX.insert(END, trueVelX)
        self.pbVelX["value"] = trueVelX + 1
        self.eVelYaw.delete(0, END)
        self.eVelYaw.insert(END, trueVelYaw)
        self.pbVelYaw["value"] = trueVelYaw + 1
        '''

        self.cmdWindow.updateView(trueVelX, trueVelYaw, 0, 0)

    def scaleIUpdated(self, event):
        if self._job:
            self.window.after_cancel(self._job)
        self._job = self.window.after(500, self.updateIByScaleI)

    def updateIByScaleI(self):
        self._job = None
        self.updateView(int(self.scaleI.get()))

    def forward(self):
        if self.i < len(self.imgAndCommandList)-1:
            self.i += 1
        else:
            self.i = 0
        self.updateView()

    def backward(self):
        if self.i > 0:
            self.i -= 1
        else:
            self.i = len(self.imgAndCommandList)-1
        self.updateView()

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

    def onClosing(self):
        #if self.player:
        #    self.player.stop()
        pass

    class CmdWindow():
        def __init__(self, master):
            self.master = master
            self.frame = Frame(self.master)
            bgColor = self.master.cget('bg')
            style = Style()
            style.theme_use('alt')
            style.configure("trueNeg.Horizontal.TProgressbar",
                            troughcolor='green', background=bgColor)
            style.configure("truePos.Horizontal.TProgressbar",
                            troughcolor=bgColor, background='green')
            style.configure("predNeg.Horizontal.TProgressbar",
                            troughcolor='blue', background=bgColor)
            style.configure("predPos.Horizontal.TProgressbar",
                            troughcolor=bgColor, background='blue')
            style.configure("trueNeg.Vertical.TProgressbar",
                            troughcolor='green', background=bgColor)
            style.configure("truePos.Vertical.TProgressbar",
                            troughcolor=bgColor, background='green')
            style.configure("predNeg.Vertical.TProgressbar",
                            troughcolor='blue', background=bgColor)
            style.configure("predPos.Vertical.TProgressbar",
                            troughcolor=bgColor, background='blue')

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

            Label(self.frame, textvariable=self.svTrueVelXPos).grid(row=2, column=3)
            Label(self.frame, textvariable=self.svPredVelXPos).grid(row=2, column=6)
            Label(self.frame, textvariable=self.svTrueVelXNeg).grid(row=7, column=3)
            Label(self.frame, textvariable=self.svPredVelXNeg).grid(row=7, column=6)
            Label(self.frame, textvariable=self.svTrueVelYawPos).grid(row=3, column=7)
            Label(self.frame, textvariable=self.svPredVelYawPos).grid(row=6, column=7)
            Label(self.frame, textvariable=self.svTrueVelYawNeg).grid(row=3, column=2)
            Label(self.frame, textvariable=self.svPredVelYawNeg).grid(row=6, column=2)

            Label(self.frame, text="green = true data").grid(row=10, column=0, columnspan=5)
            Label(self.frame, text="blue = predicted data").grid(row=10, column=6, columnspan=5)

            self.frame.pack()

        def updateView(self, trueVelX, trueVelYaw, predVelX, predVelYaw):
            self.pbTrueVelXPos["value"] = trueVelX if trueVelX > 0.0 else 0
            self.pbTrueVelXNeg["value"] = 1.0 + trueVelX if trueVelX < 0.0 else 1
            self.pbPredVelXPos["value"] = predVelX if predVelX > 0.0 else 0
            self.pbPredVelXNeg["value"] = 1.0 + predVelX if predVelX < 0.0 else 1
            self.pbTrueVelYawPos["value"] = trueVelYaw if trueVelYaw > 0.0 else 0
            self.pbTrueVelYawNeg["value"] = 1.0 + trueVelYaw if trueVelYaw < 0.0 else 1
            self.pbPredVelYawPos["value"] = predVelYaw-predVelYa if predVelYaw > 0 else 0
            self.pbPredVelYawNeg["value"] = 1.0 + predVelYaw if predVelYaw < 0.0 else 1

            trueVelXPosText = str(round(trueVelX*100)) + " %" if trueVelX > 0 else ""
            trueVelXNegText = str(round(trueVelX*100)) + " %" if trueVelX < 0 else ""
            trueVelYawPosText = str(round(trueVelYaw*100)) + " %" if trueVelYaw > 0 else ""
            trueVelYawNegText = str(round(trueVelYaw*100)) + " %" if trueVelYaw <  0 else ""
            predVelXPosText = str(round(predVelX*100)) + " %" if predVelX > 0 else ""
            predVelXNegText = str(round(predVelX*100)) + " %" if predVelX < 0 else ""
            predVelYawPosText = str(round(predVelYaw*100)) + " %" if predVelYaw > 0 else ""
            predVelYawNegText = str(round(predVelYaw*100)) + " %" if predVelYaw < 0 else ""

            self.svTrueVelXPos.set(trueVelXPosText)
            self.svPredVelXPos.set(predVelXPosText)
            self.svTrueVelXNeg.set(trueVelXNegText)
            self.svPredVelXNeg.set(predVelXNegText)
            self.svTrueVelYawPos.set(trueVelYawPosText)
            self.svPredVelYawPos.set(predVelYawPosText)
            self.svTrueVelYawNeg.set(trueVelYawNegText)
            self.svPredVelYawNeg.set(predVelYawNegText)

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
    recordingsFolder = os.path.join(os.path.expanduser("~"), "recordings")
    imgAndCommandList = inputFunctions.getImgAndCommandList(recordingsFolder, filter="left_rect")
    app = ImgAndCmdWindow(imgAndCommandList)
