import tensorflow as tf


def build_model(input_tensor=None, flags=None):
    if input_tensor is None:
        input_shape = (flags.height, flags.width, 3)
    else:
        input_shape = tuple(input_tensor.get_shape().as_list()[1:4])

    mobnet_basic = tf.keras.applications.MobileNet(include_top=False,
                                                   input_shape=input_shape,
                                                   input_tensor=input_tensor)

    # Extend mobile net by own fully connected layer
    # TODO Evaluate which activation function works best
    x = mobnet_basic.layers[-1].output
    predictions = tf.keras.layers.Dense(1, activation='linear', name='predictions')(x)
    mobnet_extended = tf.keras.Model(inputs=input_tensor, outputs=predictions,
                                     name='mobnet_extended')

    return mobnet_extended


if __name__ == "__main__":

    build_model(input_pl)
