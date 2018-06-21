import argparse
import glob
import importlib
import json
import os

import matplotlib
import progressbar
import csv

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from matplotlib import style
from tabulate import tabulate

from inputFunctions import for_subfolders_in, load_img_ctrl_pairs, ImageBatchGenerator


def plot_ref_pred_comparison(reference, predictions=None, filter=None):
    plt.figure()
    plt.title("Control command comparison")

    if filter is not None:
        plt.suptitle(filter)

    # Filter datapoints without predictions
    if predictions is not None:
        predictions = [p if p != None else np.nan for p in predictions]
        predictions = np.ma.array(predictions)
        predictions = np.ma.masked_where(predictions == np.nan, predictions)

    plt.subplot(211)
    plt.title("Yaw Velocity")
    plt.plot(range(len(reference)), reference, label="Reference", color='xkcd:orange', linewidth=2)

    if predictions is not None:
        plt.plot(range(len(predictions)), predictions, label="Prediction", color='xkcd:sky blue', linewidth=2)

    plt.grid(color='gray', linestyle='-', linewidth='1')
    plt.legend()

    plt.subplot(212)

    if predictions is not None:
        both = np.concatenate([np.asmatrix(reference), np.asmatrix(predictions)], axis=0).transpose()
        plt.hist(both, bins=11, orientation='vertical', histtype='bar', color=['xkcd:orange', 'xkcd:sky blue'],
                 label=["Reference", "Prediction"])
    else:
        plt.hist(reference, bins=11, orientation='vertical', histtype='bar', color='xkcd:orange',
                 label=["Reference", "Prediction"])

    plt.legend()

    plt.show()


class PlotLearning(Callback):
    def __init__(self):
        self.x = []
        self.values = {}
        style.use('ggplot')
        self.fig = None
        self.ax = None

    def on_train_begin(self, logs=None):
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        plt.title("Learning Progress")

    def append(self, epoch, logs=None):
        if logs is not None:
            self.x.append(epoch + 1)

            for m in self.params["metrics"]:
                if m not in self.values:
                    self.values[m] = []

                self.values[m].append(logs.get(m))

    def replot(self):
        self.ax.clear()
        self.ax.set_title("Loss History")

        num_metrics = len(self.values.keys())
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0.2, 0.8, num_metrics)]

        for metric, color in zip(self.values.keys(), colors):
            linestyle = "--" if "val" in metric else "-"
            self.ax.plot(self.x, self.values[metric], label=metric, color=color, linewidth=2, linestyle=linestyle)

        self.ax.legend()
        self.ax.grid(True)

        plt.figure(self.fig.number)
        plt.show(block=False)
        plt.draw()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        super(PlotLearning, self).on_epoch_end(epoch, logs)
        self.append(epoch, logs)

        if len(self.x) > 0:
            self.replot()

    def on_train_end(self, logs=None):
        if len(self.x) > 0:
            plt.show()


def plot_learning_curve(data_table, show_plot=True, fig_path=None):
    style.use('ggplot')
    matplotlib.rc('font', **{'family': 'Roboto', 'weight': 'normal', 'size': 10})
    fig = plt.figure()
    plt.title("Validation Loss")

    cmap = plt.get_cmap('gnuplot')
    num_runs = len(list(set(data_table["run"])))
    colors = [cmap(i) for i in np.linspace(0.2, 0.8, num_runs)]

    data_dict = {}

    for row in data_table:
        if row["run"] not in data_dict:
            data_dict[row["run"]] = {"epoch": [], "loss": [], "metric": []}

        data_dict[row["run"]]["epoch"].append(row["epoch"])
        data_dict[row["run"]]["loss"].append(row["loss"])
        data_dict[row["run"]]["metric"].append(row["metric"])

    for run, color in zip(data_dict.keys(), colors):
        plt.plot(data_dict[run]["epoch"], data_dict[run]["loss"], label=run, color=color, linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True)

    if fig_path is not None:
        fig.savefig(fig_path, dpi=100)

    if show_plot:
        plt.show()



def prepare_learning_curve_plot(args):
    assert len(args.run) > 0, "You have to specify run folders through --run argument!"

    data_table = []
    data_row = lambda x: np.array(x, dtype=[("run", "<U128"), ("epoch", np.int),
                                            ("loss", np.float32), ("metric", np.float32)])

    for run in progressbar.progressbar(args.run):
        weight_files = sorted(glob.glob(os.path.join(args.run_dir, run, "weights*.hdf5")))
        arg_file = os.path.join(args.run_dir, run, "arguments.txt")

        train_module = importlib.import_module("trainTensorFlow")

        arg_list = []

        with open(arg_file, "r", encoding="utf8") as f:
            for line in f:
                arg, value = line.strip().split("=")
                arg_list.append("--" + arg)
                arg_list.append(value)

        train_args = train_module.parser.parse_args(arg_list)

        assert train_args.model_file is not None

        model, helper = train_module.build_model(train_args.model_file, train_args, for_training=True)

        val_gen = ImageBatchGenerator(os.path.join(args.data_dir, args.val_dir), batch_size=5, shuffle=False,
                                      augment=0, crop=train_args.crop, take_or_skip=0,
                                      preprocess_input=helper.preprocess_input,
                                      preprocess_target=helper.preprocess_target)

        for weight_file in progressbar.progressbar(weight_files):
            epoch = os.path.basename(weight_file).split("_")[1]
            model.load_weights(weight_file)
            loss, metric = model.evaluate_generator(val_gen, steps=1, workers=4, use_multiprocessing=True, verbose=1)

            data_table.append(data_row((run, int(epoch), loss, metric)))

    data_arr = np.asarray(data_table)
    np.savetxt(os.path.join(args.run_dir, "{}.csv".format(args.output_file)), data_arr, fmt=['%s', '%d', '%f', '%f'],
               delimiter=",", header=",".join(data_arr.dtype.names), encoding="utf8")
    print(tabulate(data_arr, headers=["run", "epoch", "loss", "metric"]))

    plot_learning_curve(data_arr, args.show_plot, os.path.join(args.run_dir, "{}.pdf".format(args.output_file)))


def prepare_comparison_plot(args):
    ref_dict = for_subfolders_in(os.path.join(args.data_dir, args.ref_dir), apply_fn=load_img_ctrl_pairs)

    print("Select one: ")
    for i, subfolder in enumerate(ref_dict.keys()):
        print("{}) {}".format(i + 1, subfolder))

    selection = input()
    filter = list(ref_dict.keys())[int(selection.strip()) - 1]
    json_file = os.path.join(args.run_dir, args.session_dir, "predictions.json")
    predictions = None

    if os.path.exists(json_file):
        with open(json_file) as f:
            predictions = json.load(f)

        # Do some checks before merging the reference and prediction values
        basenames = [p["fileName"] + p["fileExt"] for p in predictions if filter in p["relFolderPath"]]
        pred_vals = [p['predVelYaw'] for p in predictions if filter in p["relFolderPath"]]
        assert len(pred_vals) == len(ref_dict[filter]["img_paths"]), "Predictions and ground truth have " \
                                                                     "to be of same length!"
        assert all(b == os.path.basename(f) for b, f in zip(basenames, ref_dict[filter]["img_paths"]))

        predictions = pred_vals

    plot_ref_pred_comparison(ref_dict[filter]["angular_z"], predictions, filter=filter)


if __name__ == '__main__':
    plot_parser = argparse.ArgumentParser("Plot the learning curve etc. for trained networks")
    plot_parser.add_argument("--method", action="store", type=str, default="comparison")
    plot_parser.add_argument("--data_dir", action="store", type=str, default="C:/Development/volksbot/"
                                                                             "autonomerVolksbot/data")
    plot_parser.add_argument("--run_dir", action="store", type=str, default="C:/Development/volksbot/"
                                                                            "autonomerVolksbot/run")
    plot_parser.add_argument("--session_dir", action="store", type=str, default="mobilenet_reg_lane_v13")
    plot_parser.add_argument("--ref_dir", action="store", type=str, default="test_course")
    plot_parser.add_argument("--run", action="append", type=str, default=[])
    plot_parser.add_argument("--val_dir", action="store", type=str, default="test_course_oldcfg")
    plot_parser.add_argument("--show_plot", action="store", type=int, default=1)
    plot_parser.add_argument("--output_file", action="store", type=str, default="learning_curves")
    args = plot_parser.parse_args()

    if args.method == "comparison":
        prepare_comparison_plot(args)
    elif args.method == "learning_curve":
        csv_file = os.path.join(args.run_dir, "{}.csv".format(args.output_file))

        if os.path.exists(csv_file):
            data_arr = np.genfromtxt(csv_file, delimiter=",", encoding="utf8", skip_header=1,
                                     dtype=[("run", "<U128"), ("epoch", np.int), ("loss", np.float32),
                                            ("metric", np.float32)])
            plot_learning_curve(data_arr, fig_path=os.path.join(args.run_dir, "{}.pdf".format(args.output_file)))
        else:
            prepare_learning_curve_plot(args)
    else:
        raise NotImplementedError("The method '{}' is not implemented".format(args.method))
