import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Reshape, Input
from PIL import Image


def build_model(flags):
    input_tensor = tf.keras.layers.Input((224, 224, 3), flags.batch_size)

    if input_tensor is None:
        input_shape = (flags.height, flags.width, 3)
    else:
        input_shape = tuple(input_tensor.get_shape().as_list()[1:4])

    mobnet_basic = tf.keras.applications.MobileNet(include_top=False,
                                                   input_shape=input_shape,
                                                   input_tensor=input_tensor)

    if flags.train_dense_only:
        # Disable training for the convolutional layers
        for index, layer in enumerate(mobnet_basic.layers):
            layer.trainable = False

            # if index < 89:
            # mobnet_basic.layers[index].trainable = False
            # print("{}#{}, trainable={}".format(index, layer.name, layer.trainable))
            # layer.trainable = False

    # Extend mobile net by own fully connected layer
    x = mobnet_basic.layers[-1].output
    x = Reshape((1, 1, 7 * 7 * 1024))(x)
    x = tf.keras.layers.Dense(1, activation='linear', name='predictions')(x)
    predictions = Reshape((1,))(x)
    mobnet_extended = tf.keras.Model(inputs=input_tensor, outputs=predictions,
                                     name='mobnet_extended')

    return mobnet_extended


def preprocess_input(path):
    # Open, crop, resize and rescale the image
    img = Image.open(path)
    img = img.crop((380, 0, 1100, 720))
    img = img.resize((224, 224), resample=Image.BILINEAR)

    return tf.keras.applications.mobilenet.preprocess_input(np.float32(img))
