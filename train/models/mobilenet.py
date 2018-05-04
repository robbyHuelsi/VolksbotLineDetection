import tensorflow as tf


def build_model(flags):
    mobnet_basic = tf.keras.applications.MobileNet(include_top=False,
                                                   input_shape=[flags.batch_size,
                                                                flags.height,
                                                                flags.width,
                                                                3])
    # TODO Add own fully connected layer here
    return mobnet_basic


if __name__ == "__main__":
    build_model()
