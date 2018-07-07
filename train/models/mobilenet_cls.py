import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications.mobilenet import MobileNet, preprocess_input
from .helper_api import HelperAPI


class MobileNetCls(HelperAPI):
    def monitor_val(self):
        return "val_acc"

    def monitor_mode(self):
        return "max"

    def preprocess_input(self, input):
        return preprocess_input(input)

    def preprocess_target(self, target):
        if isinstance(target, list):
            result = np.asarray([oneHotEncode(getVelYawClas(t)) for t in target])
        else:
            result = oneHotEncode(getVelYawClas(target))

        return result

    def postprocess_output(self, output):
        cls = np.argmax(output, axis=1)
        #ctrl_values = [-0.375, -0.125, 0.000, 0.125, 0.375]	## 5 classes
        ctrl_values = [-0.4375, -0.3125, -0.1875, -0.0625, 0.0000, 0.0625, 0.1875, 0.3125, 0.4375]	## 9 classes
        #ctrl_values = [-0.475, -0.425, -0.375, -0.325, -0.275, -0.225, -0.175, -0.125, -0.075, -0.025, 0.000, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475]	## 21 classes
        result = [ctrl_values[c] for c in cls]

        return result

    def build_model(self, args=None, for_training=True):
        input_shape = (224, 224, 3)
        input_tensor = Input(input_shape)
        num_classes = 9	###############################################################################################################################

        weights = None if for_training and not args.pretrained else 'imagenet'
        mobnet_basic = MobileNet(weights=weights, include_top=False, input_shape=input_shape, input_tensor=input_tensor)

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

        # Extend mobile net by own fully connected layer
        x = mobnet_basic.layers[-1].output

        # TODO Evaluate if the original layers from MobileNet bring better performance/accuracy
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 1024), name='reshape_1')(x)
        x = Dropout(0.5, name='dropout')(x)
        x = Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        predictions = Flatten()(x)

        # x = Flatten()(x)
        # predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

        mobnet_extended = Model(inputs=input_tensor, outputs=predictions, name='mobilenet_cls')

        # Finalize the model by compiling it
        if for_training:
            mobnet_extended.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                                    optimizer=Adam(lr=args.learning_rate, decay=args.decay_rate))

        return mobnet_extended


def getVelYawClas(avVelYaw, minYaw=-0.5, maxYaw=0.5, classes=9):	###################################################################################
    avVelYaw = np.clip(avVelYaw, minYaw, maxYaw)

    #if minYaw <= avVelYaw < minYaw / 2:	velYawClass = 0
    #if minYaw / 2 <= avVelYaw < -0.001:	velYawClass = 1
    #if -0.001 <= avVelYaw <= 0.001:		velYawClass = 2
    #if 0.001 < avVelYaw <= maxYaw / 2:	velYawClass = 3
    #if maxYaw / 2 < avVelYaw <= maxYaw:	velYawClass = 4

    # TODO Is there a smarter way to do it?
    if minYaw <= avVelYaw < 3 * minYaw / 4:			velYawClass = 0
    if 3 * minYaw / 4 <= avVelYaw < 2 * minYaw / 4:	velYawClass = 1
    if 2 * minYaw / 4 <= avVelYaw < minYaw / 4:		velYawClass = 2
    if minYaw / 4 <= avVelYaw < -0.001:				velYawClass = 3
    if -0.001 <= avVelYaw <= 0.001:					velYawClass = 4
    if 0.001 < avVelYaw <= maxYaw / 4:				velYawClass = 5
    if maxYaw / 4 < avVelYaw <= 2 * maxYaw / 4:		velYawClass = 6
    if 2 * maxYaw / 4 < avVelYaw <= 3 * maxYaw / 4:	velYawClass = 7
    if 3 * maxYaw / 4 < avVelYaw <= maxYaw:			velYawClass = 8

    #if minYaw <= avVelYaw < 9 * minYaw / 10:			velYawClass = 0
    #if 9 * minYaw / 10 <= avVelYaw < 8 * minYaw / 10:	velYawClass = 1
    #if 8 * minYaw / 10 <= avVelYaw < 7 * minYaw / 10:	velYawClass = 2
    #if 7 * minYaw / 10 <= avVelYaw < 6 * minYaw / 10:	velYawClass = 3
    #if 6 * minYaw / 10 <= avVelYaw < 5 * minYaw / 10:	velYawClass = 4
    #if 5 * minYaw / 10 <= avVelYaw < 4 * minYaw / 10:	velYawClass = 5
    #if 4 * minYaw / 10 <= avVelYaw < 3 * minYaw / 10:	velYawClass = 6
    #if 3 * minYaw / 10 <= avVelYaw < 2 * minYaw / 10:	velYawClass = 7
    #if 2 * minYaw / 10 <= avVelYaw < minYaw / 10:		velYawClass = 8
    #if minYaw / 10 <= avVelYaw < -0.001:				velYawClass = 9
    #if -0.001 <= avVelYaw <= 0.001:					velYawClass = 10
    #if 0.001 < avVelYaw <= maxYaw / 10:				velYawClass = 11
    #if maxYaw / 10 < avVelYaw <= 2 * maxYaw / 10:		velYawClass = 12
    #if 2 * maxYaw / 10 < avVelYaw <= 3 * maxYaw / 10:	velYawClass = 13
    #if 3 * maxYaw / 10 < avVelYaw <= 4 * maxYaw / 10:	velYawClass = 14
    #if 4 * maxYaw / 10 < avVelYaw <= 5 * maxYaw / 10:	velYawClass = 15
    #if 5 * maxYaw / 10 < avVelYaw <= 6 * maxYaw / 10:	velYawClass = 16
    #if 6 * maxYaw / 10 < avVelYaw <= 7 * maxYaw / 10:	velYawClass = 17
    #if 7 * maxYaw / 10 < avVelYaw <= 8 * maxYaw / 10:	velYawClass = 18
    #if 8 * maxYaw / 10 < avVelYaw <= 9 * maxYaw / 10:	velYawClass = 19
    #if 9 * maxYaw / 10 < avVelYaw <= maxYaw:			velYawClass = 20
    return velYawClass


def oneHotEncode(velYawClass, classes=9):	###########################################################################################################
    encoded = to_categorical(velYawClass, num_classes=classes)
    return encoded


model_helper = MobileNetCls()
