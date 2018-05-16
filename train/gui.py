import os
import json
from tkinter import *
from PIL import ImageTk, Image


# View and Controll
class imgAndCmdWindow():
    def __init__(self, msg, imgPath, velX, velYaw):

        self.inputbox.title(msg)
        # self.inputbox.state("zoomed")

        imgFrame = Image.open(imgPath)
        imgFrameResized = imgFrame.resize((800, 600), Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(imgFrameResized)
        lFrame = Label(self.inputbox, image=itkFrame)
        lFrame.grid(row=0, columnspan=3)

        lVelX = Label(self.inputbox, text="Vel. X")
        self.eVelX = Entry(self.inputbox)
        self.eVelX.insert(END, velX)
        lVelX.grid(row=2, sticky="e")
        self.eVelX.grid(row=2, column=1, columnspan=2)
        self.eVelX.focus_set()

        lVelYaw = Label(self.inputbox, text="Vel. Yaw")
        self.eVelVaw = Entry(self.inputbox)
        self.eVelVaw.insert(END, eVelVaw)
        lVelYaw.grid(row=3, sticky="e")
        self.eCin.grid(row=3, column=1, columnspan=2)

        bForward = Button(self.inputbox, text="Forward",
                          width=10, command=self._bDoneClicked)
        bForward.grid(row=6, column=1)

        bBackward = Button(self.inputbox, text="Backward",
                           width=10, command=self._bNextClicked)
        bBackward.grid(row=6, column=2)

        self.inputbox.mainloop()

    def _buttonClicked(self, status):
        self.labelId = self.svLabelId.get()
        self.bin = self.eBin.get()
        self.cin = self.eCin.get()
        self.ean = self.eEan.get()
        self.quantity = self.eQuantity.get()
        self.status = status
        self.inputbox.destroy()

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
    testPath = os.path.join(os.path.expanduser("~"), "recordings",
                            "2018-04-23_12-02-26", "1524477746744130648.jpg")
    window1 = imgAndCmdWindow("hello!", testPath, "1", "2")
