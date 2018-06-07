import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, Reshape, Conv2D, Activation, Conv1D
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet, preprocess_input
from PIL import Image

from models.helper_api import HelperAPI


class MobileNetReg(HelperAPI):
    def monitor_val(self):
        return "val_loss"

    def monitor_mode(self):
        return "min"

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
        shape = (1, 1, 1024)

        x = mobnet_basic.layers[-1].output
        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(0.5, name='dropout')(x)
        # x = Conv2D(512, (1, 1), padding='same', activation='relu')(x)
        # x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
        x = Conv2D(49, (1, 1), padding='same', name='pre_predictions', activation='relu')(x)
        x = Conv2D(1, (1, 1), padding='same', name='predictions', activation='linear')(x)
        predictions = Flatten()(x)
        mobnet_extended = Model(inputs=input_tensor, outputs=predictions, name='mobilenet_reg')

        # Finalize the model by compiling it
        if for_training:
            mobnet_extended.compile(loss='mae', metrics=['mse'],
                                    optimizer=Adam(lr=args.learning_rate, decay=args.decay_rate))

        return mobnet_extended

    def preprocess_input(self, input, crop=True):
        # Open, (crop,) resize and rescale the image
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
