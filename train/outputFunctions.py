import glob
import json
import os
import numpy as np


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
