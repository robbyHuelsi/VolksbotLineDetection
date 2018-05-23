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
        self.window.title(str(self.imgAndCommandList[self.i]["imgPath"]))
        self._job = None  # Necessary for changing scale value by hand
        self.player = None

        # self.inputbox.state("zoomed")

        self.windowWidth = 1200

        img = Image.open(self.imgAndCommandList[self.i]["imgPath"])
        img = img.resize((self.windowWidth, int(img.size[1]/img.size[0]*self.windowWidth)), Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(img)
        self.lImgFrame = Label(self.window, image=itkFrame)
        self.lImgFrame.grid(row=0, columnspan=3)

        self.scaleI = Scale(self.window, from_=0, to=len(self.imgAndCommandList), length=self.windowWidth, orient=HORIZONTAL, command=self.scaleIUpdated)
        self.scaleI.grid(row=1, columnspan=3)

        lVelX = Label(self.window, text="Vel. X")
        self.eVelX = Entry(self.window)
        self.eVelX.insert(END, self.imgAndCommandList[self.i]["velX"])
        lVelX.grid(row=2, column=0)
        self.eVelX.grid(row=2, column=1)
        self.eVelX.focus_set()
        self.pbVelX = Progressbar(self.window, orient=HORIZONTAL,
                                      length=200, mode="determinate")
        self.pbVelX["maximum"] = 2
        self.pbVelX["value"] = 1
        self.pbVelX.grid(row=2, column=2)


        lVelYaw = Label(self.window, text="Vel. Yaw")
        self.eVelYaw = Entry(self.window)
        self.eVelYaw.insert(END, self.imgAndCommandList[self.i]["velYaw"])
        lVelYaw.grid(row=3, column=0)
        self.eVelYaw.grid(row=3, column=1,)
        self.pbVelYaw = Progressbar(self.window, orient=HORIZONTAL,
                                      length=200, mode="determinate")
        self.pbVelYaw["maximum"] = 2
        self.pbVelYaw["value"] = 1
        self.pbVelYaw.grid(row=3, column=2)

        bBackward = Button(self.window, text="Backward",
                           width=10, command=self._bBackwardClicked)
        bBackward.grid(row=4, column=0)

        bForward = Button(self.window, text="Forward",
                          width=10, command=self._bForwardClicked)
        bForward.grid(row=4, column=2)

        self.bPlayStop = Button(self.window, text="Play",
                                width=10, command=self._bPlayPausedClicked)
        self.bPlayStop.grid(row=4, column=1)

        self.cmdWindow = ImgAndCmdWindow.CmdWindow(Toplevel(self.window))

        # self.window.protocol("WM_DELETE_WINDOW", self.onClosing())
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

        self.eVelX.delete(0, END)
        self.eVelX.insert(END, trueVelX)
        self.pbVelX["value"] = trueVelX + 1
        self.eVelYaw.delete(0, END)
        self.eVelYaw.insert(END, trueVelYaw)
        self.pbVelYaw["value"] = trueVelYaw + 1

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

            style = Style()
            style.theme_use('clam')
            style.configure("trueNeg.Horizontal.TProgressbar",
                            troughcolor='green', background='gray')
            style.configure("truePos.Horizontal.TProgressbar",
                            troughcolor='gray', background='green')
            style.configure("predNeg.Horizontal.TProgressbar",
                            troughcolor='blue', background='gray')
            style.configure("predPos.Horizontal.TProgressbar",
                            troughcolor='gray', background='blue')
            style.configure("trueNeg.Vertical.TProgressbar",
                            troughcolor='green', background='gray')
            style.configure("truePos.Vertical.TProgressbar",
                            troughcolor='gray', background='green')
            style.configure("predNeg.Vertical.TProgressbar",
                            troughcolor='blue', background='gray')
            style.configure("predPos.Vertical.TProgressbar",
                            troughcolor='gray', background='blue')

            self.pbTrueVelXPos = Progressbar(self.frame, orient=VERTICAL,
                                             length=200, mode="determinate",
                                             style="truePos.Vertical.TProgressbar")
            self.pbTrueVelXPos["maximum"] = 1
            self.pbTrueVelXPos["value"] = 0
            self.pbTrueVelXPos.grid(row=0, column=1)

            self.pbPredVelXPos = Progressbar(self.frame, orient=VERTICAL,
                                             length=200, mode="determinate",
                                             style="predPos.Vertical.TProgressbar")
            self.pbPredVelXPos["maximum"] = 1
            self.pbPredVelXPos["value"] = 0
            self.pbPredVelXPos.grid(row=0, column=2)

            self.pbTrueVelYawNeg = Progressbar(self.frame, orient=HORIZONTAL,
                                               length=200, mode="determinate",
                                               style="trueNeg.Horizontal.TProgressbar")
            self.pbTrueVelYawNeg["maximum"] = 1
            self.pbTrueVelYawNeg["value"] = 0
            self.pbTrueVelYawNeg.grid(row=1, column=0)

            self.pbPredVelYawNeg = Progressbar(self.frame, orient=HORIZONTAL,
                                               length=200, mode="determinate",
                                               style="predNeg.Horizontal.TProgressbar")
            self.pbPredVelYawNeg["maximum"] = 1
            self.pbPredVelYawNeg["value"] = 0
            self.pbPredVelYawNeg.grid(row=2, column=0)

            self.pbTrueVelYawPos = Progressbar(self.frame, orient=HORIZONTAL,
                                               length=200, mode="determinate",
                                               style="truePos.Horizontal.TProgressbar")
            self.pbTrueVelYawPos["maximum"] = 1
            self.pbTrueVelYawPos["value"] = 0
            self.pbTrueVelYawPos.grid(row=1, column=3)

            self.pbPredVelYawPos = Progressbar(self.frame, orient=HORIZONTAL,
                                               length=200, mode="determinate",
                                               style="predPos.Horizontal.TProgressbar")
            self.pbPredVelYawPos["maximum"] = 1
            self.pbPredVelYawPos["value"] = 0
            self.pbPredVelYawPos.grid(row=2, column=3)

            self.pbTrueVelXNeg = Progressbar(self.frame, orient=VERTICAL,
                                             length=200, mode="determinate",
                                             style="trueNeg.Vertical.TProgressbar")
            self.pbTrueVelXNeg["maximum"] = 1
            self.pbTrueVelXNeg["value"] = 0
            self.pbTrueVelXNeg.grid(row=3, column=1)

            self.pbPredVelXNeg = Progressbar(self.frame, orient=VERTICAL,
                                             length=200, mode="determinate",
                                             style="predNeg.Vertical.TProgressbar")
            self.pbPredVelXNeg["maximum"] = 1
            self.pbPredVelXNeg["value"] = 0
            self.pbPredVelXNeg.grid(row=3, column=2)

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
