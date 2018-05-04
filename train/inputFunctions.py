import os


def getImgAndCommandList(recordingsFolder):
    # TODO Think about how the input_fn loads images and controls lazy and
    # controlled way. Number of epochs is defined, number of iterations depends
    # on the batch size and number of samples in the training set.

    cmdVelFiles = []
    imgsFolders = {}
    outputList = []

    for directory, dirnames, filenames in os.walk(recordingsFolder):
        if directory == recordingsFolder:
            for f in filenames:
                fName, fExtension = os.path.splitext(f)
                if fExtension == ".csv":
                    cmdVelFiles.append(fName)
        else:
            # print directory
            # tmpImgs = []
            # for f in filenames:
            imgsFolders[directory] = filenames

    # print cmdVelFiles
    for imgFolder, imgFiles in imgsFolders.iteritems():
        print str(imgFiles)


if __name__ == "__main__":
    recordingsFolder = os.path.join(os.path.expanduser("~"), "recordings")
    getImgAndCommandList(recordingsFolder)
