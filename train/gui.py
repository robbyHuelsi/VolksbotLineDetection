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


        #self.window.protocol("WM_DELETE_WINDOW", self.onClosing())
        self.window.mainloop()

    def updateView(self, i=-1):
        if i > -1:
            self.i = i
            
        self.window.title(str(self.imgAndCommandList[self.i]["imgPath"]))

        img = Image.open(self.imgAndCommandList[self.i]["imgPath"])
        img = img.resize((self.windowWidth, int(img.size[1]/img.size[0]*self.windowWidth)), Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(img)
        self.lImgFrame.configure(image=itkFrame)
        self.lImgFrame.image = itkFrame
        
        self.scaleI.set(self.i)

        self.eVelX.delete(0, END)
        self.eVelX.insert(END, self.imgAndCommandList[self.i]["velX"])
        self.pbVelX["value"] = self.imgAndCommandList[self.i]["velX"] +1
        self.eVelYaw.delete(0, END)
        self.eVelYaw.insert(END, self.imgAndCommandList[self.i]["velYaw"])
        self.pbVelYaw["value"] = self.imgAndCommandList[self.i]["velYaw"] +1
        
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
