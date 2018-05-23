import glob
import tensorflow as tf
import importlib
import datetime
import os
import numpy as np
import argparse
from inputFunctions import ImageBatchGenerator


parser = argparse.ArgumentParser(description='Train and predict on different models')

# Hyperparameters of the training run
parser.add_argument("--model_file", action="store", default=None, type=str,
                    help="Relative path to either a python module in models directory or "
                         "hdf5-saved model file.")
parser.add_argument("--train_dense_only", action="store", default=False, type=bool,
                    help="Whether or not to train the dense layer only.")
parser.add_argument("--augment_train_data", action="store", default=False, type=bool,
                    help="Whether or not to augment the training data.")
parser.add_argument("--epochs", action="store", default=10, type=int,
                    help="Repeat the training over all samples the number of epochs.")
parser.add_argument("--batch_size", action="store", default=32, type=int,
                    help="Number of training samples in one batch.")
parser.add_argument("--learning_rate", action="store", default=1e-5, type=float,
                    help="Learning rate of the specified optimizer.")
parser.add_argument("--decay_rate", action="store", default=5e-6, type=float,
                    help="Decay the learning rate by this value after one epoch.")

# Session specific directories, paths and files
parser.add_argument("--session_name", action="store", default=str(datetime.datetime.now()).
                    replace(" ", "_").replace(":", "-")[:-7], type=str,
                    help="Session name of this training and validation run.")
parser.add_argument("--run_dir", action="store", default="/tmp/run", type=str,
                    help="Parent directory of the <session_name> folder.")
parser.add_argument("--save_file", action="store", default="checkpoint.hdf5", type=str,
                    help="Name of the file where the model and weights get saved.")

# Dataset specific parameters
parser.add_argument("--data_dir", action="store", default=os.path.join(
                    os.path.expanduser("~"), "recordings"), type=str,
                    help="Path to the dataset directory.")
parser.add_argument("--train_dir", action="store", default="train", type=str,
                    help="Subdirectory in dataset directory for training images")
parser.add_argument("--val_dir", action="store", default="val", type=str,
                    help="Subdirectory in dataset directory for validation images.")
# TODO Remove, in my opinion this dataset splitting is a bad idea
parser.add_argument("--split_ind", action="store", default=6992, type=int,
                    help="Index where the data set will be split into training and "
                         "validation set.")
parser.add_argument("--img_filter", action="store", default="left_rect", type=str,
                    help="Choose which camera image is used as network input.")

# Tensorflow specific parameters
parser.add_argument("--log_level", action="store", default=tf.logging.INFO, type=int,
                    help="Verbosity of tensorflow and the script.")
parser.add_argument("--seed", action="store", default=0, type=int,
                    help="Set the random seed of numpy and tensorflow to this value.")


def build_model(model_file, args=None):
    # Build the model depending on the flags either from keras model definition in code
    # or restore the model from hdf5 file
    if model_file and os.path.exists(os.path.join("models", "{}.py".format(model_file))):
        # Import the model from a python module/code
        module = importlib.import_module("models.{}".format(model_file))
        helper = module.model_helper
        model = helper.build_model(args)
    elif model_file and os.path.exists(model_file):
        # Restore the keras model from a hdf5 file
        helper = None
        model = tf.keras.models.load_model(model_file)
    else:
        raise ValueError("Model file '{}' does not exist!".format(model_file))

    return model, helper


def restore_weights(model, save_file):
    if os.path.exists(save_file):
        # If the model directory and the save file already exist, try to recover already saved weights
        tf.logging.info("Restoring weights from {}!".format(save_file))
        model.load_weights(save_file)
    else:
        raise ValueError("Save file {} does not exist!".format(save_file))


def predict(model, helper, img_paths=None, pred_gen=None):
    if pred_gen is None and img_paths is None:
        raise ValueError("Either 'pred_gen' or 'img_paths' have to be supplied!")

    if img_paths is None:
        output = model.predict_generator(pred_gen, workers=4, use_multiprocessing=True, verbose=1)
    else:
        output = model.predict(np.expand_dims(helper.preprocess_input(img_paths), axis=0), verbose=1)

    predictions = helper.postprocess_output(output)

    return predictions


def save_arguments(argument_file, args):
    with open(argument_file, "w") as fh:
        for arg in vars(args):
            fh.write("{}={}\n".format(arg, getattr(args, arg)))


def save_predictions(img_paths, predictions, path):
    assert len(img_paths) == len(predictions)

    with open(path, "w") as fh:
        for prediction, img_path in zip(predictions, img_paths):
            path, file_name = os.path.split(img_path)
            dirs = path.split(os.sep)
            rel_path = os.path.join(dirs[-2], dirs[-1], file_name)

            fh.write("{},{}\n".format(rel_path, prediction[0][0]))


def main(args):
    # Fixate the random seeds of numpy and Tensorflow is the first thing to do
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Set the logging level of Tensorflow
    tf.logging.set_verbosity(args.log_level)

    # Define keras model as None initially
    model, helper = build_model(args.model_file, args)

    # Output for TensorBoard and model file will be inside args.tb_dir
    save_dir = os.path.join(args.run_dir, "{}_{}".format(model.name, args.session_name))
    save_file = os.path.join(save_dir, args.save_file)

    # Create a model specific directory where the weights are saved
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Restore the weights, existence of save_file is checked inside the function
    restore_weights(model, save_file)

    # If the number of epochs is greater zero, training cycles are run
    if args.epochs > 0:
        # The image batch generator that handles the image loading
        train_gen = ImageBatchGenerator(os.path.join(args.data_dir, args.train_dir), batch_size=args.batch_size,
                                        preprocess_input=helper.preprocess_input,
                                        preprocess_target=helper.preprocess_target,
                                        sub_dir=args.img_filter)
        val_gen = ImageBatchGenerator(os.path.join(args.data_dir, args.val_dir), batch_size=args.batch_size,
                                      preprocess_input=helper.preprocess_input,
                                      preprocess_target=helper.preprocess_target,
                                      sub_dir=args.img_filter)

        # TODO Think about adding early stopping as callback here
        # TODO Add plotting callback https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
        ms_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
            monitor='val_loss', verbose=1, save_best_only=False,
            save_weights_only=True, mode='auto', period=1)

        # Fit the model to the data by previously defined conditions (optimizer, loss ...)
        model.fit_generator(generator=train_gen, epochs=args.epochs, shuffle=True, workers=4,
                            validation_data=val_gen, callbacks=[ms_callback])

        # Save the current model weights and used arguments
        save_arguments(os.path.join(save_dir, "arguments.txt"), args)
        model.save(save_file)
        tf.logging.info('Saved final model and weights to {}!'.format(save_file))

    # TODO Change later! Right now the predictions are made for all training images
    pred_gen = ImageBatchGenerator(os.path.join(args.data_dir, args.train_dir),
                                   batch_size=args.batch_size,
                                   preprocess_input=helper.preprocess_input,
                                   preprocess_target=helper.preprocess_target,
                                   sub_dir=args.img_filter)

    predictions = predict(model, helper, pred_gen=pred_gen)
    save_predictions(pred_gen.features, predictions, os.path.join(save_dir, "predictions.txt"))


if __name__ == "__main__":
    main(parser.parse_args())
