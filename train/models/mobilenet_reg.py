import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, GlobalAveragePooling2D, Reshape, Conv2D
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.regularizers import l1

from .helper_api import HelperAPI


class MobileNetReg(HelperAPI):
    def monitor_val(self):
        return "val_loss"

    def monitor_mode(self):
        return "min"

    def build_model(self, args=None, for_training=True):
        input_shape = (224, 224, 3)
        input_tensor = Input(input_shape)

        weights = None if for_training and not args.pretrained else 'imagenet'
        mobnet_basic = MobileNet(weights=weights, include_top=False, input_shape=input_shape, input_tensor=input_tensor)

        if for_training and args.train_dense_only:
            # Disable training for the convolutional layers
            for index, layer in enumerate(mobnet_basic.layers):
                layer.trainable = False

        reg = l1(args.regularize) if for_training else l1(0.0)
        dropout = args.dropout if for_training else 0.5

        # Extend mobilenet by own fully connected layer
        x = mobnet_basic.layers[-1].output
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 1024), name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(49, (1, 1), padding='same', name='pre_predictions', activation='relu',
                   kernel_regularizer=reg, bias_regularizer=reg)(x)
        x = Conv2D(1, (1, 1), padding='same', name='predictions', activation='linear',
                   kernel_regularizer=reg, bias_regularizer=reg)(x)
        predictions = Flatten()(x)

        mobnet_extended = Model(inputs=input_tensor, outputs=predictions, name='mobilenet_reg')

        # Finalize the model by compiling it
        if for_training:
            mobnet_extended.compile(loss='mae', metrics=['mse'],
                                    optimizer=Adam(lr=args.learning_rate, decay=args.decay_rate))

        return mobnet_extended

    def preprocess_input(self, input):
        return preprocess_input(input)

    def preprocess_target(self, target):
        return target

    def postprocess_output(self, output):
        output = np.clip(output, -0.5, 0.5)

        # Absolute zeroing of small values
        greater = np.greater_equal(-0.001, output)
        less = np.less_equal(0.001, output)
        between = greater & less
        output[between] = 0.0

        return output


model_helper = MobileNetReg()
