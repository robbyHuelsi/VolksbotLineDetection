import os
import json
from tkinter import *
from PIL import ImageTk, Image

import inputFunctions

# View and Controll
class imgAndCmdWindow():
    def __init__(self, imgAndCommandList):
        this.imgAndCommandList = imgAndCommandList
        this.i = 0
        window = Tk()
        window.title(str(this.imgAndCommandList[this.i]["imgPath"]))

        # self.inputbox.state("zoomed")

        imgFrame = Image.open(this.Img)
        imgFrameResized = imgFrame.resize((600, 600), Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(imgFrameResized)
        lFrame = Label(window, image=itkFrame)
        lFrame.grid(row=0, columnspan=3)

        lVelX = Label(window, text="Vel. X")
        self.eVelX = Entry(window)
        self.eVelX.insert(END, velX)
        lVelX.grid(row=2, sticky="e")
        self.eVelX.grid(row=2, column=1, columnspan=2)
        self.eVelX.focus_set()

        lVelYaw = Label(window, text="Vel. Yaw")
        self.eVelYaw = Entry(window)
        self.eVelYaw.insert(END, velYaw)
        lVelYaw.grid(row=3, sticky="e")
        self.eVelYaw.grid(row=3, column=1, columnspan=2)

        bForward = Button(window, text="Forward",
                          width=10, command=self._bForwardClicked)
        bForward.grid(row=6, column=1)

        bBackward = Button(window, text="Backward",
                           width=10, command=self._bBackwardClicked)
        bBackward.grid(row=6, column=2)

        window.mainloop()

    def _buttonClicked(self, status):
        self.labelId = self.svLabelId.get()
        self.bin = self.eBin.get()
        self.cin = self.eCin.get()
        self.ean = self.eEan.get()
        self.quantity = self.eQuantity.get()
        self.status = status

    def _bForwardClicked(self):
        self._buttonClicked("forward")

    def _bBackwardClicked(self):
        self._buttonClicked("backward")


'''
    def getInfo(self):
        #self.inputbox.wait_window()
        return self.status, self.labelId, self.VelX, self.VelYaw
'''


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"), "recordings")
    imgAndCommandList = inputFunctions.getImgAndCommandList(recordingsFolder)
    app = imgAndCmdWindow(imgAndCommandList)
