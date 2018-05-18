import os
import json
from tkinter import *
from PIL import ImageTk, Image
import time
from threading import Thread

import inputFunctions

# View and Controll
class imgAndCmdWindow():
    def __init__(self, imgAndCommandList):
        self.imgAndCommandList = imgAndCommandList
        self.i = 0
        self.window = Tk()
        self.window.title(str(self.imgAndCommandList[self.i]["imgPath"]))
        self.player = self.Player(self)

        # self.inputbox.state("zoomed")

        imgFrame = Image.open(self.imgAndCommandList[self.i]["imgPath"])
        # imgFrame = imgFrame.resize((600, 600), Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(imgFrame)
        self.lImgFrame = Label(self.window, image=itkFrame)
        self.lImgFrame.grid(row=0, columnspan=3)

        lVelX = Label(self.window, text="Vel. X")
        self.eVelX = Entry(self.window)
        self.eVelX.insert(END, self.imgAndCommandList[self.i]["velX"])
        lVelX.grid(row=1, sticky="e")
        self.eVelX.grid(row=1, column=1, columnspan=2)
        self.eVelX.focus_set()

        lVelYaw = Label(self.window, text="Vel. Yaw")
        self.eVelYaw = Entry(self.window)
        self.eVelYaw.insert(END, self.imgAndCommandList[self.i]["velYaw"])
        lVelYaw.grid(row=2, sticky="e")
        self.eVelYaw.grid(row=2, column=1, columnspan=2)

        bBackward = Button(self.window, text="Backward",
                           width=10, command=self._bBackwardClicked)
        bBackward.grid(row=3, column=0)

        bForward = Button(self.window, text="Forward",
                          width=10, command=self._bForwardClicked)
        bForward.grid(row=3, column=2)

        self.bPlayStop = Button(self.window, text="Play",
                          width=10, command=self._bPlayPausedClicked)
        self.bPlayStop.grid(row=3, column=1)

        self.window.mainloop()

    def updateView(self):
        self.window.title(str(self.imgAndCommandList[self.i]["imgPath"]))

        imgFrame = Image.open(self.imgAndCommandList[self.i]["imgPath"])
        # imgFrame = imgFrame.resize((600, 600), Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(imgFrame)
        self.lImgFrame.configure(image=itkFrame)
        self.lImgFrame.image = itkFrame

        self.eVelX.delete(0, END)
        self.eVelX.insert(END, self.imgAndCommandList[self.i]["velX"])
        self.eVelYaw.delete(0, END)
        self.eVelYaw.insert(END, self.imgAndCommandList[self.i]["velYaw"])

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
        if self.player.isPlaying():
            self.player.stop()
            self.bPlayStop.config(text="Play")
        else:
            self.player.play()
            self.bPlayStop.config(text="Stop")

    class Player(Thread):
        def __init__(self, imgAndCmdWindow):
            self.imgAndCmdWindow = imgAndCmdWindow
            self.playing = False

        def play(self):
            self.playing = True
            self.run()

        def stop(self):
            self.playing = False

        def isPlaying(self):
            return self.playing

        def run(self):
            if self.playing:
                self.imgAndCmdWindow.forward()
                time.sleep(0.5)
                #self.run()



'''
    def getInfo(self):
        #self.inputbox.wait_window()
        return self.status, self.labelId, self.VelX, self.VelYaw
'''


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"), "recordings")
    imgAndCommandList = inputFunctions.getImgAndCommandList(recordingsFolder)
    app = imgAndCmdWindow(imgAndCommandList)
