import tensorflow as tf
import importlib
import datetime
import os
import numpy as np
import tqdm
from inputFunctions import create_dataset


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
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Number of training samples in one batch.")
tf.app.flags.DEFINE_integer("width", 224,
                            "Width of the images that enter the network.")
tf.app.flags.DEFINE_integer("height", 224,
                            "Height of the images that enter the network.")
tf.app.flags.DEFINE_string("feature_key", "imgPath",
                           "The feature key that ")
FLAGS = tf.app.flags.FLAGS


def main(argvs=None):
    # Set the logging level of tensorflow
    tf.logging.set_verbosity(FLAGS.log_level)

    # Import the keras model file dynamically if specified or exit otherwise
    if FLAGS.keras_model is None:
        tf.logging.warn("Keras model file has to be specified!")
        exit(1)
    else:
        keras_model_module = \
            importlib.import_module("models.{}".format(FLAGS.keras_model))

    # Fixate the random seeds of numpy and tensorflow
    np.random.seed(0)
    tf.set_random_seed(0)

    # Create the tensorflow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # TODO How to load checkpoints, saved models from previous runs?
    # TODO How to do the training exactly
    dataset, iterations = create_dataset(FLAGS.dataset_dir, FLAGS.epochs, FLAGS.batch_size)

    iterator = dataset.make_one_shot_iterator()
    next_img, next_ctrl = iterator.get_next()

    input_tensor = tf.keras.Input((224, 224, 3), FLAGS.batch_size)
    model = keras_model_module.build_model(input_tensor, FLAGS)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='mean_squared_error')

    # Output for TensorBoard and model file will be inside FLAGS.tb_dir
    save_dir = os.path.join(FLAGS.tb_dir, "{}_{}".format(model.name,
                                                         FLAGS.session_name))
    tf.keras.callbacks.TensorBoard(log_dir=save_dir, histogram_freq=1,
                                   batch_size=FLAGS.batch_size, write_graph=True,
                                   write_grads=True, write_images=False)

    t_epochs = tqdm.trange(FLAGS.epochs, desc='Epochs')
    t_iters = tqdm.trange(iterations, desc='Iterations')

    for _ in t_epochs:
        for _ in t_iters:
            x_batch, y_batch = sess.run([next_img, next_ctrl])
            loss = model.train_on_batch(x_batch, y_batch)
            t_iters.set_description("Loss: {}".format(np.round(loss, 4)))
            t_iters.refresh()

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model.save(save_dir)
    tf.logging.info('Saved trained model at {} '.format(save_dir))

    # Load the test dataset
    x_test, y_test = None

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    tf.logging.info('Test loss: {}'.format(scores[0]))
    tf.logging.info('Test accuracy: {}'.format(scores[1]))


if __name__ == "__main__":
    # TensorFlow like way to run the main function
    tf.app.run()
