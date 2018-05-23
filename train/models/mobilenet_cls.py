import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten
from tensorflow.python.keras.applications import MobileNet
from PIL import Image


def build_model():
    input_shape = (224, 224, 3)
    input_tensor = Input(input_shape)
    num_classes = 9

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

    # Extend mobile net by own fully connected layer
    x = mobnet_basic.layers[-1].output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

    mobnet_extended = Model(inputs=input_tensor, outputs=predictions, name='mobilenet_cls')

    # Finalize the model by compiling it
    mobnet_extended.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                            optimizer=tf.keras.optimizers.Adam(lr=flags.learning_rate, decay=flags.decay_rate))

    return mobnet_extended


def preprocess_input(path):
    # Open, crop, resize and rescale the image
    img = Image.open(path)
    img = img.crop((380, 0, 1100, 720))
    img = img.resize((224, 224), resample=Image.BILINEAR)

    return tf.keras.applications.mobilenet.preprocess_input(np.float32(img))


def preprocess_target(target):
    if isinstance(target, list):
        result = np.asarray([oneHotEncode(getVelYawClas(t)) for t in target])
    else:
        result = oneHotEncode(getVelYawClas(target))

    return result


def postprocess_output(output):
    cls = np.argmax(output, axis=1)
    ctrl_values = [-0.875, -0.625, -0.375, -0.1255, 0, 0.1255, 0.375, 0.625, -0.875]
    result = [ctrl_values[c] for c in cls]

    return result


def getVelYawClas(avVelYaw, minYaw=-1, maxYaw=1, classes=9):
    avVelYaw = np.clip(avVelYaw, minYaw, maxYaw)

    # TODO Is there a smarter way to do it?
    if minYaw <= avVelYaw < 3 * minYaw / 4:        velYawClass = 0
    if 3 * minYaw / 4 <= avVelYaw < minYaw / 2:    velYawClass = 1
    if minYaw / 2 <= avVelYaw < minYaw / 4:        velYawClass = 2
    if minYaw / 4 <= avVelYaw < -0.001:        velYawClass = 3
    if -0.001 <= avVelYaw <= 0.001:            velYawClass = 4
    if 0.001 < avVelYaw <= maxYaw / 4:        velYawClass = 5
    if maxYaw / 4 < avVelYaw <= maxYaw / 2:        velYawClass = 6
    if maxYaw / 2 < avVelYaw <= 3 * maxYaw / 4:    velYawClass = 7
    if 3 * maxYaw / 4 < avVelYaw <= maxYaw:        velYawClass = 8

    return velYawClass


def oneHotEncode(velYawClass, classes=9):
    encoded = tf.keras.utils.to_categorical(velYawClass, num_classes=classes)
    return encoded


if __name__ == "__main__":
    print(preprocess_target(20))