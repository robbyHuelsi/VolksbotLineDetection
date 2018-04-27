import os
from os.path import expanduser


def main():
    recordingsFolder = os.path.join(expanduser("~"), "recordings")      # Get recordings folder

    cmdVelFiles = []
    imgsFolders = {}

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

    # print imgsFolders


if __name__ == "__main__":
    main()
