import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten
from tensorflow.python.keras.applications import MobileNet
from PIL import Image


def build_model(flags):
    input_shape = (224, 224, 3)
    input_tensor = Input(input_shape, flags.batch_size)

    mobnet_basic = MobileNet(include_top=False, input_shape=input_shape, input_tensor=input_tensor)

    if flags.train_dense_only:
        # Disable training for the convolutional layers
        for index, layer in enumerate(mobnet_basic.layers):
            layer.trainable = False

            # TODO Remove code below if unnecessary
            # Disable the training for some layers of mobilenet
            # if index < 89:
            # mobnet_basic.layers[index].trainable = False
            # print("{}#{}, trainable={}".format(index, layer.name, layer.trainable))
            # layer.trainable = False

    # Extend mobilenet by own fully connected layer
    x = mobnet_basic.layers[-1].output
    x = Flatten()(x)
    predictions = Dense(1, activation='linear', name='predictions')(x)
    mobnet_extended = Model(inputs=input_tensor, outputs=predictions, name='mobilenet_reg')

    # Finalize the model by compiling it
    mobnet_extended.compile(loss='mean_absolute_error', metrics=['mse'],
                            optimizer=tf.keras.optimizers.Adam(lr=flags.learning_rate, decay=flags.decay_rate))

    return mobnet_extended


def preprocess_input(path):
    # Open, crop, resize and rescale the image
    img = Image.open(path)
    img = img.crop((380, 0, 1100, 720))
    img = img.resize((224, 224), resample=Image.BILINEAR)

    return tf.keras.applications.mobilenet.preprocess_input(np.float32(img))


def preprocess_target(target):
    return target


def postprocess_output(output):
    return np.clip(output, -1.0, 1.0)
