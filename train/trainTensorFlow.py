import tensorflow as tf
import importlib
import datetime
import os
import numpy as np


from inputFunctions import ImageBatchGenerator


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
tf.app.flags.DEFINE_string("save_file", "checkpoint.hdf5",
                           "Name of the file where the model and weights get saved.")
tf.app.flags.DEFINE_integer("split_ind", 6992,
                            "Index where the data set will be split into training and validation set.")
tf.app.flags.DEFINE_float("learning_rate", 0.00001,
                          "Learning rate of the specified optimizer.")
tf.app.flags.DEFINE_string("restore_file", None,
                           "Path to model file that should be restored. This overrides the keras_model flag.")
FLAGS = tf.app.flags.FLAGS


def main(argvs=None):
    # Set the logging level of tensorflow
    tf.logging.set_verbosity(FLAGS.log_level)

    # Define keras model as None initially
    model = None

    # Build the model depending on the set flags either from keras model definition in code
    # or restore the model from hdf5 file
    if FLAGS.keras_model:
        # Import the keras model file dynamically if specified or exit otherwise
        keras_model_module = importlib.import_module("models.{}".format(FLAGS.keras_model))
        model = keras_model_module.build_model(FLAGS)
    elif FLAGS.restore_file and os.path.exists(FLAGS.restore_file):
        # Restore the keras model from a restore file
        model = tf.keras.models.load_model(FLAGS.restore_file)

    if model is None:
        tf.logging.error("Model creation failed, process is exiting now!")
        tf.logging.error("Check the Flags: keras_model={}, restore_file={}".format(FLAGS.keras_model,
                                                                                   FLAGS.restore_file))
        exit(1)

    # Fixate the random seeds of numpy and tensorflow
    np.random.seed(0)
    tf.set_random_seed(0)

    # Finalize the model by compiling it
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate, decay=5e-6), loss='mean_absolute_error',
                  metrics=['mse'])

    # Output for TensorBoard and model file will be inside FLAGS.tb_dir
    save_dir = os.path.join(FLAGS.tb_dir, "{}_{}".format(model.name, FLAGS.session_name))
    save_file = os.path.join(save_dir, FLAGS.save_file)

    # Create a model specific directory where the weights are saved
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    elif os.path.exists(save_file):
        # If the model directory and the save file already exist, try to recover already saved weights
        tf.logging.info("Restoring weights from {}!".format(save_file))
        model.load_weights(save_file)

    # Tensorboard callback
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir, batch_size=FLAGS.batch_size, write_graph=True,
                                                 histogram_freq=0,  # Setting this to 1 will produce a failure at
                                                                    # training procedure end
                                                 write_grads=False,  # Setting this to True will produce a failure at
                                                                     # training procedure start
                                                 write_images=False)  # Not necessary
    ms_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
                                                     monitor='val_loss', verbose=1, save_best_only=False,
                                                     save_weights_only=True, mode='auto', period=1)

    # The image batch generator that handles the image loading
    train_gen = ImageBatchGenerator(FLAGS.dataset_dir, batch_size=FLAGS.batch_size, dim=(FLAGS.height, FLAGS.width),
                                    end_ind=FLAGS.split_ind)
    val_gen = ImageBatchGenerator(FLAGS.dataset_dir, batch_size=FLAGS.batch_size, dim=(FLAGS.height, FLAGS.width),
                                  start_ind=FLAGS.split_ind)

    # Fit the model to the data by previously defined conditions (optimizer, loss ...)
    model.fit_generator(generator=train_gen, epochs=FLAGS.epochs, shuffle=True, workers=4,
                        validation_data=val_gen, callbacks=[tb_callback, ms_callback])

    # Save the current model weights
    model.save(save_file)
    tf.logging.info('Saved final model and weights to {}!'.format(save_file))

    # Use the validation generator for model evaluation
    if len(val_gen) > 0:
        scores = model.evaluate_generator(val_gen, workers=4, use_multiprocessing=True)
        tf.logging.info('Final loss: {}'.format(scores[0]))


if __name__ == "__main__":
    # TensorFlow like way to run the main function
    tf.app.run()
