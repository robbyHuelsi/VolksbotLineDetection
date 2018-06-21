import json
import os
import datetime
import tensorflow as tf


def save_arguments(argument_file, args):
    with open(argument_file, "w") as fh:
        for arg in vars(args):
            fh.write("{}={}\n".format(arg, getattr(args, arg)))


def save_predictions(img_paths, predictions, json_path):
    assert len(img_paths) == len(predictions)

    predictions_list = []

    for prediction, img_path in zip(predictions, img_paths):
        prediction_dict = {}
        path, file_name_ext = os.path.split(img_path)
        file_name, file_ext = os.path.splitext(file_name_ext)
        dirs = path.split(os.path.sep)
        rel_path = os.path.join(dirs[-2], dirs[-1])
        prediction_dict["relFolderPath"] = rel_path
        prediction_dict["fileName"] = file_name
        prediction_dict["fileExt"] = file_ext

        try:
            _ = iter(prediction)
        except TypeError:
            prediction_dict["predVelYaw"] = float(prediction)
        else:
            prediction_dict["predVelYaw"] = float(prediction[0])

        predictions_list.append(prediction_dict)

    if predictions_list:
        with open(json_path, 'w') as fp:
            json.dump(predictions_list, fp)


def join_path(parts, none_check=True):
    return None if None in parts and none_check else os.path.join(*parts)


def avoid_override(file_path, prefix_timestamp=True):
    if os.path.exists(file_path):
        basename = os.path.basename(file_path)

        if prefix_timestamp:
            timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")[:-7]
            new_basename = "{}_{}".format(timestamp, basename)
            new_path = file_path.replace(basename, new_basename)
            tf.logging.warning("File '{}' already exists, new file named '{}' to avoid override!".format(basename,
                                                                                                         new_basename))
        else:
            bak_path = file_path + ".bak"
            bak_basename = os.path.basename(bak_path)
            new_path = file_path
            os.rename(file_path, bak_basename)
            tf.logging.warning("File '{}' already exists, backed it up as '{}' to avoid override!".format(basename,
                                                                                                          bak_basename))

        return new_path
    else:
        return file_path
