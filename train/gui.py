import os
import json
from tkinter import *
from PIL import ImageTk,Image
from builtins import str


class annotation():
    def __init__(self):
        pass
    
    def annotate(self, pathToImages, pathForRealData, pathToMasterData, onlyNotAnnotated = False):
        with open(pathToMasterData) as data_file:
            masterData = json.loads(data_file.read())
             
        for imageFileName in os.listdir(pathToImages):
            filename, extension = os.path.splitext(imageFileName)
            if extension == ".jpg":
                imageFilePath = os.path.join(pathToImages, imageFileName)
                readoutDataFilePath = os.path.join(pathToImages, imageFileName + ".json")
                realDataFilePath = os.path.join(pathForRealData, imageFileName + ".json")
                labelId = ""
                bin = ""
                cin = ""
                ean = ""
                quantity = ""
                
                if os.path.isfile(realDataFilePath):
                    if onlyNotAnnotated:
                        print("Only not annotated. " + realDataFilePath + " already exists.")
                        continue
                    
                    with open(realDataFilePath) as realdataFile:
                        realdataJson = json.loads(realdataFile.read())
                        try:
                            labelId = realdataJson[""]
                        except:
                            pass
                        try:
                            bin = realdataJson[""]
                        except:
                            pass
                        try:
                            cin = realdataJson[""]
                        except:
                            pass
                        try:
                            ean = realdataJson[""]
                        except:
                            pass
                        try:
                            quantity = realdataJson[""]
                        except:
                            pass
                el

class inputboxRealData():
    def __init__(self):
        
        self.inputbox.title(msg)
        self.inputbox.state("zoomed")
        
        imgFrame = Image.open(imgPath)
        imgFrameResized = imgFrame.resize((800, 600),Image.ANTIALIAS)
        itkFrame = ImageTk.PhotoImage(imgFrameResized)
        lFrame = Label(self.inputbox, image = itkFrame)
        lFrame.grid(row=0, columnspan = 3)
        
        lLabelId = Label(self.inputbox, text = "")
        self.svLabelId = StringVar(self.inputbox)
        self.svLabelId.set(self.labelId) # default value
        omLabelId = OptionMenu(self.inputbox, self.svLabelId, "one", "two", "three")
        omLabelId.config(state = DISABLED)
        lLabelId.grid(row = 1, sticky = "e")
        omLabelId.grid(row = 1, column = 1, columnspan = 2)
        #omLabelId.focus_set()
        
        lBin = Label(self.inputbox, text = "")
        self.eBin = Entry(self.inputbox)
        self.eBin.insert(END, self.bin)
        lBin.grid(row = 2, sticky = "e")
        self.eBin.grid(row = 2, column = 1, columnspan = 2)
        self.eBin.focus_set()
        
        lCin = Label(self.inputbox, text = "")
        self.eCin = Entry(self.inputbox)
        self.eCin.insert(END, self.cin)
        lCin.grid(row = 3, sticky = "e")
        self.eCin.grid(row = 3, column = 1, columnspan = 2)
        
        lEan = Label(self.inputbox, text = "")
        self.eEan = Entry(self.inputbox)
        self.eEan.insert(END, self.ean)
        lEan.grid(row = 4, sticky = "e")
        self.eEan.grid(row = 4, column = 1, columnspan = 2)
        
        lQuantity = Label(self.inputbox, text = "")
        self.eQuantity = Entry(self.inputbox)
        self.eQuantity.insert(END, self.quantity)
        lQuantity.grid(row = 5, sticky = "e")
        self.eQuantity.grid(row = 5, column = 1, columnspan = 2)
        
        bDone = Button(self.inputbox, text = "Save & next", width = 10, command = self._bDoneClicked)
        bDone.grid(row = 6, column = 1)
        
        bNext = Button(self.inputbox, text = "Skip", width = 10, command = self._bNextClicked)
        bNext.grid(row = 6, column = 2)
        
        self.inputbox.mainloop()
    
    
    def _buttonClicked(self, status):
        self.labelId = self.svLabelId.get()
        self.bin = self.eBin.get()
        self.cin = self.eCin.get()
        self.ean = self.eEan.get()
        self.quantity = self.eQuantity.get()
        self.status = status
        self.inputbox.destroy()
        
    def _bDoneClicked(self):
        self._buttonClicked("done")
        
    def _bNextClicked(self):
        self._buttonClicked("next")
        
    def getInfo(self):
        #self.inputbox.wait_window()
        return self.status, self.labelId, self.bin, self.cin, self.ean, self.quantity        


if __name__ == "__main__":
    pathToImages = os.path.join("..", "data", "readout_2017-12-05")
    pathForRealData = os.path.join("..", "data", "readout_2017-12-05_realData")
    pathToMasterData = os.path.join("..", "data", "masterData", "master_data_selection_2017-11-15.json")
    onlyNotAnnotated = True
    annotation().annotate(pathToImages, pathForRealData, pathToMasterData, onlyNotAnnotated)