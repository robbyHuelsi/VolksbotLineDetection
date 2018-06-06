import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.applications import MobileNet
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet, preprocess_input
from PIL import Image

from models.helper_api import HelperAPI


class MobileNetReg(HelperAPI):
    def build_model(self, args=None, for_training=True):
        input_shape = (224, 224, 3)
        input_tensor = Input(input_shape)

        mobnet_basic = MobileNet(include_top=False, input_shape=input_shape, input_tensor=input_tensor)

        if for_training and args.train_dense_only:
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
        x = Dense(9, activation='relu')(x)
        predictions = Dense(1, activation='linear', name='predictions')(x)
        mobnet_extended = Model(inputs=input_tensor, outputs=predictions, name='mobilenet_reg')

        # Finalize the model by compiling it
        if for_training:
            mobnet_extended.compile(loss='mean_absolute_error', metrics=['mse'],
                                    optimizer=Adam(lr=args.learning_rate, decay=args.decay_rate))

        return mobnet_extended

    def preprocess_input(self, input, crop=True):
        # Open, crop, resize and rescale the image
        img = Image.open(input)

        if crop:
            img = img.crop((380, 0, 1100, 720))

        img = img.resize((224, 224), resample=Image.NEAREST)  # , resample=Image.BILINEAR)

        return preprocess_input(np.float32(img))

    def preprocess_target(self, target):
        return target

    def postprocess_output(self, output):
        return np.clip(output, -1.0, 1.0)


model_helper = MobileNetReg()
