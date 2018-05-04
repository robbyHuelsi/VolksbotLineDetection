import tensorflow as tf
import importlib
import datetime
import os
import numpy as np
import glob
from inputFunctions import getImgAndCommandList

tf.app.flags.DEFINE_string("session_name", str(datetime.datetime.now()).
                           replace(" ", "_").replace(":", "-")[:-7],
                           "Session name of this training/eval/prediction run.")
tf.app.flags.DEFINE_integer("log_level", tf.logging.INFO,
                            "Verbosity of tensorflow and the script")
tf.app.flags.DEFINE_string("dataset_dir", os.path.join(os.path.expanduser("~"),
                                                       "recordings"),
                           "Path to the dataset directory.")
tf.app.flags.DEFINE_string("tb_dir", "/tmp/tb/",
                           "Path to the tensorboard directory.")
tf.app.flags.DEFINE_string("keras_model", None,
                           "Path to keras model file.")
tf.app.flags.DEFINE_integer("epochs", 10,
                            "Training runs for this number of epochs.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Number of training samples in one batch.")
tf.app.flags.DEFINE_integer("width", 224,
                            "Width of the images that enter the network.")
tf.app.flags.DEFINE_integer("height", 224,
                            "Height of the images that enter the network.")
FLAGS = tf.app.flags.FLAGS


def create_dataset(epochs, batch_size):
    # filename, controls = getImgAndCommandList(FLAGS.dataset_dir)
    filenames = glob.glob("/home/florian/recordings/2018-04-23_12-13-51/*.jpg")
    controls = list(range(len(filenames)))

    dataset = tf.data.Dataset.from_tensor_slices((filenames, controls))
    dataset = dataset.map(load_img)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(128)

    return dataset, int(np.ceil(len(filenames) / batch_size))


def load_img(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_bilinear(image_decoded, [FLAGS.height, FLAGS.width])

    return image_resized, label


def augment_image(filename, label):
    # TODO Implement later if needed (e.g. random gaussian noise, random left-right flip
    return filename, label


def main(argvs=None):
    # Set the logging level of tensorflow
    tf.logging.set_verbosity(FLAGS.log_level)

    # Fixate the random seeds of numpy and tensorflow
    np.random.seed(0)
    tf.set_random_seed(0)

    # Create the tensorflow session
    sess = tf.Session()

    # TODO How to load checkpoints, saved models from previous runs?
    # TODO How to do the training exactly
    dataset, iterations = create_dataset(FLAGS.epochs, FLAGS.batch_size)

    iterator = dataset.make_one_shot_iterator()
    next = iterator.get_next()

    input_pl = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    output_pl = tf.placeholder(dtype=tf.float32, shape=[], name='output')

    # Import the keras model file dynamically if specified or exit otherwise
    if FLAGS.keras_model is None:
        tf.logging.warn("Keras model file has to be specified!")
        exit(1)
    else:
        keras_model_module = \
            importlib.import_module("models.{}".format(FLAGS.keras_model))

    model = keras_model_module.build_model(input_pl, FLAGS)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9,
                                                     beta_2=0.999, epsilon=None,
                                                     decay=0.0, amsgrad=False),
                  loss='mean_squared_error')

    # Output for TensorBoard and model file will be inside FLAGS.tb_dir
    save_dir = os.path.join(FLAGS.tb_dir, "{}_{}".format(model.name,
                                                         FLAGS.session_name))

    # model.fit(data, labels, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)

    for epoch in range(FLAGS.epochs):
        for iteration in range(iterations):
            x_batch, y_batch = sess.run(next)
            model.train_on_batch(x_batch, y_batch)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model.save(save_dir)
    tf.logging.info('Saved trained model at {} '.format(save_dir))

    # Load the test dataset
    x_test, y_test = False

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    tf.logging.info('Test loss: {}'.format(scores[0]))
    tf.logging.info('Test accuracy: {}'.format(scores[1]))


if __name__ == "__main__":
    # TensorFlow like way to run the main function
    tf.app.run()
