import tensorflow as tf
import importlib
import datetime
import os
import numpy as np
from inputFunctions import ImageBatchGenerator


# Hyperparameters of the training run
tf.app.flags.DEFINE_string("model_file", None,
                           "Relative path to either a python module in models directory or hdf5-saved model file.")
tf.app.flags.DEFINE_boolean("train_dense_only", False,
                            "Whether or not to train the dense layer only.")
tf.app.flags.DEFINE_boolean("augment_train_data", False,
                            "Whether or not to augment the training data.")
tf.app.flags.DEFINE_integer("epochs", 10,
                            "Repeat the training over all samples the number of epochs.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Number of training samples in one batch.")
tf.app.flags.DEFINE_float("learning_rate", 1e-5,
                          "Learning rate of the specified optimizer.")
tf.app.flags.DEFINE_float("decay_rate", 5e-6,
                          "Decay the learning rate by this value after one epoch.")

# Session specific directories, paths and files
tf.app.flags.DEFINE_string("session_name", str(datetime.datetime.now()).
                           replace(" ", "_").replace(":", "-")[:-7],
                           "Session name of this training and validation run.")
tf.app.flags.DEFINE_string("run_dir", "/tmp/tb",
                           "Parent directory of the <session_name> folder.")
tf.app.flags.DEFINE_string("save_file", "checkpoint.hdf5",
                           "Name of the file where the model and weights get saved.")

# Dataset specific parameters
tf.app.flags.DEFINE_string("data_dir", os.path.join(os.path.expanduser("~"), "recordings"),
                           "Path to the dataset directory.")
# TODO Remove, in my opinion this dataset splitting is a bad idea
tf.app.flags.DEFINE_integer("split_ind", 6992,
                            "Index where the data set will be split into training and validation set.")
tf.app.flags.DEFINE_string("img_filter", "left_rect",
                           "Choose which camera image is used as network input.")

# Tensorflow specific parameters
tf.app.flags.DEFINE_integer("log_level", tf.logging.INFO,
                            "Verbosity of tensorflow and the script.")

tf.app.flags
FLAGS = tf.app.flags.FLAGS


def main(argvs=None):
    # Fixate the random seeds of numpy and Tensorflow is the first thing to do
    np.random.seed(0)
    tf.set_random_seed(0)

    # Set the logging level of Tensorflow
    tf.logging.set_verbosity(FLAGS.log_level)

    # Define keras model as None initially
    model = None
    preprocess_input_fn = None

    # Build the model depending on the flags either from keras model definition in code
    # or restore the model from hdf5 file
    if FLAGS.model_file and os.path.exists(os.path.join("models", "{}.py".format(FLAGS.model_file))):
        # Import the model from a python module/code
        module = importlib.import_module("models.{}".format(FLAGS.model_file))
        model = module.build_model(FLAGS)
        preprocess_input_fn = module.preprocess_input
    elif FLAGS.model_file and os.path.exists(FLAGS.model_file):
        # Restore the keras model from a hdf5 file
        model = tf.keras.models.load_model(FLAGS.model_file)
    else:
        tf.logging.error("Model file '{}' does not exist!".format(FLAGS.model_file))
        exit(1)

    # Finalize the model by compiling it
    model.compile(loss='mean_absolute_error', metrics=['mse'],
                  optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate, decay=5e-6))

    # Output for TensorBoard and model file will be inside FLAGS.tb_dir
    save_dir = os.path.join(FLAGS.run_dir, "{}_{}".format(model.name, FLAGS.session_name))
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
    train_gen = ImageBatchGenerator(os.path.join(FLAGS.data_dir, "train"), batch_size=FLAGS.batch_size,
                                    preprocess_input_fn=preprocess_input_fn, img_filter=FLAGS.img_filter)
    val_gen = ImageBatchGenerator(os.path.join(FLAGS.data_dir, "val"), batch_size=FLAGS.batch_size,
                                  preprocess_input_fn=preprocess_input_fn, img_filter=FLAGS.img_filter)

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
