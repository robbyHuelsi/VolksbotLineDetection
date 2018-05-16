import tensorflow as tf
from tensorflow.python.keras.layers import Reshape


def build_model(input_tensor, flags):
    if input_tensor is None:
        input_shape = (flags.height, flags.width, 3)
    else:
        input_shape = tuple(input_tensor.get_shape().as_list()[1:4])

    mobnet_basic = tf.keras.applications.MobileNet(include_top=False,
                                                   input_shape=input_shape,
                                                   input_tensor=input_tensor)

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
    predictions = Reshape((1, ))(x)
    mobnet_extended = tf.keras.Model(inputs=input_tensor, outputs=predictions,
                                     name='mobnet_extended')

    return mobnet_extended
