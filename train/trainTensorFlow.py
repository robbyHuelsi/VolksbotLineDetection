import tensorflow as tf
import importlib
import datetime

tf.app.flags.DEFINE_string("session_name", str(datetime.datetime.now()).
                           replace(" ", "_").replace(":", "-")[:-7],
                           "Session name of this training/eval/prediction run.")
tf.app.flags.DEFINE_integer("log_level", tf.logging.INFO,
                            "Verbosity of tensorflow and the script")
tf.app.flags.DEFINE_string("dataset_dir", os.path.join(expanduser("~"),
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

        model = keras_model_module.build_model()

    # TODO Loading of the test samples
    x_test, y_test = None

    # Output for TensorBoard and model file will be inside FLAGS.tb_dir
    save_dir = os.path.join(FLAGS.tb_dir, "{}_{}".format(model.name,
                                                         FLAGS.session_name))

    # TODO How to load checkpoints, saved models from previous runs?
    # TODO How to do the training exactly

    # If the input function is implemented as Keras data generator
    # model.fit_generator(generator.flow(x_train, y_train,
    #                                    batch_size=FLAGS.batch_size),
    #                     epochs=FLAGS.epochs,
    #                     validation_data=(x_test, y_test),
    #                     workers=4)

    # If the input function is simple
    # model.train_on_batch(x_batch, y_batch)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model.save(save_dir)
    tf.logging.info('Saved trained model at {} '.format(save_dir))

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    tf.logging.info('Test loss: {}'.format(scores[0]))
    tf.logging.info('Test accuracy: {}'.format(scores[1]))


if __name__ == "__main__":
    # TensorFlow like way to run the main function
    tf.app.run()
