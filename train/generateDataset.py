import glob
import shutil
from PIL import Image
from PIL import ImageEnhance
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.mobilenet import preprocess_input
import progressbar


# def pillow_augmentations(pil_img, sharpness=2.5, contrast=0.25, color=2.5, brightness=0.25):
def pillow_augmentations(pil_img, sharpness=1.0, contrast=0.25, color=1.0, brightness=0.25):
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(np.random.normal(loc=1.0, scale=sharpness))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(np.random.normal(loc=1.0, scale=contrast))
    pil_img = ImageEnhance.Color(pil_img).enhance(np.random.normal(loc=1.0, scale=color))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(np.random.normal(loc=1.0, scale=brightness))

    return pil_img


def gaussian_noise(np_img, gaussian_noise=0.02):
    return np_img + np.random.normal(loc=0.0, scale=gaussian_noise, size=np_img.shape)


def channel_wise_zero_mean(np_img):
    return np_img - np.mean(np.mean(np_img, axis=0, keepdims=True), axis=1, keepdims=True)


def zero_mean(np_img):
    return np_img - np.mean(np_img)


def generate_dataset(recordings_folder, target_folder, type=None, std=0.0):
    assert type in ["brightness", "color", "contrast", "sharpness"]

    lst_dir = None

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    for dirpath, dirnames, filenames in progressbar.progressbar(os.walk(recordings_folder)):
        if dirpath == recordings_folder:
            for filename in filenames:
                if not filename.endswith(".csv"):
                    continue

                src_file = os.path.join(recordings_folder, filename)
                dst_file = os.path.join(target_folder,
                                        os.path.splitext(filename)[0] + '_{}_{:.2f}.csv'.format(type, std))
                shutil.copyfile(src_file, dst_file)
        else:
            if dirnames:
                head, tail = os.path.split(dirpath)
                lst_dir = os.path.join(target_folder, tail + '_{}_{:.2f}'.format(type, std))
                os.mkdir(lst_dir)
            else:
                assert lst_dir, "lst_dir should not be 'None'"
                head, tail = os.path.split(dirpath)
                dst_dir = os.path.join(lst_dir, tail)
                os.mkdir(dst_dir)

                if filenames:
                    for filename in progressbar.progressbar(filenames):
                        if not str(filename).endswith(".jpg"):
                            continue

                        src_file = os.path.join(dirpath, filename)
                        dst_file = os.path.join(dst_dir, filename)

                        src_img = Image.open(src_file)

                        if type == "brightness":
                            dst_img = ImageEnhance.Brightness(src_img).enhance(np.random.normal(loc=1.0, scale=std))
                        elif type == "contrast":
                            dst_img = ImageEnhance.Contrast(src_img).enhance(np.random.normal(loc=1.0, scale=std))
                        elif type == "sharpness":
                            dst_img = ImageEnhance.Sharpness(src_img).enhance(np.random.normal(loc=1.0, scale=std))
                        elif type == "color":
                            dst_img = ImageEnhance.Color(src_img).enhance(np.random.normal(loc=1.0, scale=std))
                        else:
                            raise NotImplementedError

                        dst_img.resize((224, 224), resample=Image.NEAREST).save(dst_file)


def visualize_augmentation(folder):
    imgs = glob.glob(os.path.join(folder, "*", "left_rect", "*.jpg"))
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
        plt.imshow((np_img + 1.0) / 2.0, vmin=0.0, vmax=1.0)
        plt.title(str(os.path.basename(img_path)))
        plt.subplot(122)
        plt.imshow(np.float32(src_img) / 255.0)

        plt.show(block=False)
        plt.waitforbuttonpress()


if __name__ == "__main__":
    folders = ["C:/Development/volksbot/autonomerVolksbot/data/train_lane",
               "C:/Development/volksbot/autonomerVolksbot/data/train_lane_curve",
               "C:/Development/volksbot/autonomerVolksbot/data/train_lane_inner_correction",
               "C:/Development/volksbot/autonomerVolksbot/data/train_lane_outer_correction"]

    visualize_augmentation(folders[0])
    exit(0)
    for folder in progressbar.progressbar(folders):
        np.random.seed(0)
        generate_dataset(folder, folder + "_aug", type="contrast", std=2.0)
        np.random.seed(0)
        generate_dataset(folder, folder + "_aug", type="brightness", std=0.25)
        np.random.seed(0)
        generate_dataset(folder, folder + "_aug", type="color", std=5.0)
        np.random.seed(0)
        generate_dataset(folder, folder + "_aug", type="sharpness", std=20.0)
