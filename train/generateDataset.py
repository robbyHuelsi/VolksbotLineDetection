import glob
import shutil
from PIL import Image
from PIL import ImageEnhance
import random
import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    imgs = glob.glob("/home/florian/Development/tmp/data/train_lane/*/left_rect/*.jpg")
    np.random.seed(0)
    src_img = Image.open(imgs[0])
    print(imgs[0])

    for img_path in imgs:
        rnds = {"sharpness": np.random.normal(loc=1.0, scale=2.5),
                "color": np.random.normal(loc=1.0, scale=2.5),
                "contrast": np.random.normal(loc=1.0, scale=0.25),
                "brightness": np.random.normal(loc=1.0, scale=0.25)}

        src_img = Image.open(img_path)

        # TODO skip 0.0 values for brightness and contrast
        # img = ImageEnhance.Sharpness(src_img).enhance(rnds["sharpness"])
        # img = ImageEnhance.Color(img).enhance(rnds["color"])
        # img = ImageEnhance.Contrast(img).enhance(rnds["contrast"])
        # img = ImageEnhance.Brightness(img).enhance(rnds["brightness"])

        # np_img = np.float32(src_img)/255.0
        # print("min/max: {}/{}".format(np.min(np_img), np.max(np_img)))
        # mean = np.mean(np_img) - 0.5
        # np_img = np_img - mean
        # print("old mean: {:.4f}, new mean: {:.4f}".format(mean, np.mean(np_img)))
        # print("min/max: {}/{}".format(np.min(np_img), np.max(np_img)))

        plt.cla()
        plt.clf()
        plt.subplot(121)
        plt.imshow(np.clip(np_img, 0.0, 1.0), vmin=0.0, vmax=1.0)
        plt.title(str(rnds))
        plt.subplot(122)
        plt.imshow(np.float32(src_img)/255.0)
        plt.show(block=False)
        plt.waitforbuttonpress()


def generateDataset(recordingsFolder, targetFolder):
    if os.path.isdir(targetFolder) == False:
        os.mkdir(targetFolder)
    for dirpath, dirnames, filenames in os.walk(recordingsFolder):
        if dirpath == recordingsFolder:
            for i in filenames:
                srcFile = os.path.join(recordingsFolder, i)
                dstFile = os.path.join(targetFolder, os.path.splitext(i)[0] + '_aug.csv')
                # dst = os.path.join(targetFolder, tail)
                shutil.copyfile(srcFile, dstFile)
        else:
            if dirnames != []:
                head, tail = os.path.split(dirpath)
                lstDir = os.path.join(targetFolder, tail + '_aug')
                os.mkdir(lstDir)
            else:
                head, tail = os.path.split(dirpath)
                # dstDir = os.path.join(targetFolder, tail)
                dstDir = os.path.join(lstDir, tail)
                os.mkdir(dstDir)
                if filenames != []:
                    for i in filenames:
                        srcFile = os.path.join(dirpath, i)
                        dstFile = os.path.join(dstDir, i)
                        # if os.path.isdir(targetFolder) == False:
                        #	os.mkdir(targetFolder)
                        srcImg = Image.open(srcFile)  # open i
                        rndm = random.randint(1, 4)
                        if rndm == 1:
                            dstImg = ImageEnhance.Sharpness(srcImg).enhance(random.randint(0, 100))
                        if rndm == 2:
                            dstImg = ImageEnhance.Color(srcImg).enhance(random.randint(0, 20))
                        if rndm == 3:
                            dstImg = ImageEnhance.Contrast(srcImg).enhance(round(random.uniform(0.100, 10.000), 3))
                        if rndm == 4:
                            dstImg = ImageEnhance.Brightness(srcImg).enhance(round(random.uniform(0.100, 10.000), 3))
                        # shutil.copy(srcfile, dstdir)	# copy i to target
                        dstImg.save(dstFile)
