import glob
import shutil
from PIL import Image
from PIL import ImageEnhance
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.mobilenet import preprocess_input


def pillow_augmentations(pil_img, sharpness=0.25, color=0.25, contrast=0.25, brightness=0.25):
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(np.random.normal(loc=1.0, scale=sharpness))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(np.random.normal(loc=1.0, scale=color))
    pil_img = ImageEnhance.Color(pil_img).enhance(np.random.normal(loc=1.0, scale=contrast))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(np.random.normal(loc=1.0, scale=brightness))

    return pil_img


def gaussian_noise(np_img, gaussian_noise=0.02):
    return np_img + np.random.normal(loc=0.0, scale=gaussian_noise, size=np_img.shape)


def channel_wise_zero_mean(np_img):
    return np_img - np.mean(np.mean(np_img, axis=0, keepdims=True), axis=1, keepdims=True)


def zero_mean(np_img):
    return np_img - np.mean(np_img)


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


if __name__ == "__main__":
    imgs = glob.glob("C:\\Development\\volksbot\\autonomerVolksbot\\data\\train_lane_curve\\*\\left_rect\\*.jpg")
    np.random.seed(0)

    for img_path in imgs:
        src_img = Image.open(img_path).resize((224, 224), Image.NEAREST)
        img = pillow_augmentations(src_img)
        np_img = preprocess_input(np.float32(img))
        np_img = gaussian_noise(np_img)
        np_img = channel_wise_zero_mean(np_img)
        np_img = np.clip(np_img, -1.0, 1.0)
        print("min/max: {}/{}".format(np.min(np_img), np.max(np_img)))
        print("{}".format(np.mean(np_img)))

        plt.cla()
        plt.clf()
        plt.subplot(121)
        plt.imshow((np_img+1.0)/2.0, vmin=0.0, vmax=1.0)
        plt.title(str(os.path.basename(img_path)))
        plt.subplot(122)
        plt.imshow(np.float32(src_img)/255.0)

        plt.show(block=False)
        plt.waitforbuttonpress()