import argparse
import glob
import importlib
import json
import os
import matplotlib
import progressbar
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from keras.callbacks import Callback
from matplotlib import style, gridspec
from tabulate import tabulate
from inputFunctions import ImageBatchGenerator, getImgAndCommandList


color = {"green":  "#85be48", "gray": "#8a8b8a", "orange": "#ffa500", "light_orange": "#ffe0b5",
         "blue": "#0fa3b1", "pink": "#6b2d5c"}
cc = itertools.cycle(color.values())


def plot_ref_pred_comparison(reference, predictions=None, filter=None, factor=0.005, start_ind=0, end_ind=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.5), sharey=True, gridspec_kw={"width_ratios": [3, 1]})
    matplotlib.rc('font', **{'weight': 'normal', 'size': 8})

    reference = np.asarray(reference) / factor
    reference = reference[start_ind:end_ind]
    dps = len(reference)

    # if filter is not None:
    #     plt.suptitle(filter)

    # Filter datapoints without predictions
    if predictions is not None:
        predictions = [p if p != None else np.nan for p in predictions]
        predictions = np.ma.array(predictions)
        predictions = np.ma.masked_where(predictions == np.nan, predictions)
        predictions = predictions / factor
        predictions = predictions[start_ind:end_ind]

    #plt.suptitle("Steuerbefehl-Vergleich")

    ax1.set_title("Steuerbefehl Vergleich - Verlauf")
    ax1.plot(range(dps), reference, label="Referenz", color=color["orange"], linewidth=2)

    if predictions is not None:
        ax1.plot(range(dps), predictions, label="Vorhersage", color=color["blue"], linewidth=2)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_ylabel("Drehgeschwindigkeit [%]")
    ax1.set_xlabel("Bild-Nummer")
    ax1.legend(fancybox=True, shadow=True, ncol=1) # loc='lower center') #, bbox_to_anchor=(0.5, 1.5))
    ax1.grid(color=color["gray"], linestyle='-', linewidth='1')

    # Plot histogram of controls
    ax2.set_title("Histogramm")

    bins = np.arange(-0.5, 0.6, 0.1) / factor

    if predictions is not None:
        both = np.concatenate([np.asmatrix(predictions), np.asmatrix(reference)], axis=0).transpose()
        ax2.hist(both, bins=bins, orientation='horizontal', histtype='step', color=[color["blue"], color["orange"]],
                 label=["Vorhersage", "Referenz"], linewidth=2)
    else:
        ax2.hist(np.asarray(reference), bins=bins, orientation='horizontal', histtype='step', color=color["orange"],
                 label="Referenz", linewidth=2)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel("Anzahl")
    #ax2.set_ylim([-40, 100])
    #ax2.set_xlabel("Drehgeschwindigkeit [%]")
    ax2.grid(color=color["gray"], linestyle='-', linewidth='1')

    fig.tight_layout()
    fig.savefig("../documentation/comp.pdf", pad_inches=0.0)
    plt.show()


class PlotLearning(Callback):
    def __init__(self, run_name, show_progress=True, plot_output_file=None, val_output_file=None):
        self.x = []
        self.values = {}
        style.use('ggplot')
        self.fig = None
        self.ax = None
        self.run_name = run_name
        self.plot_output_file = plot_output_file
        self.val_output_file = val_output_file
        self.show_progress = show_progress

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
        self.ax.set_title("Loss/Metric Training History")
        plt.suptitle('Run: {}'.format(self.run_name))

        num_metrics = len(self.values.keys())
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0.2, 0.8, num_metrics)]
        marker = None if len(self.x) > 1 else "x"

        for metric, color in zip(self.values.keys(), colors):
            linestyle = "--" if "val" in metric else "-"
            self.ax.plot(self.x, self.values[metric], label=metric, color=color, linewidth=2, linestyle=linestyle,
                         marker=marker)

        self.ax.legend()
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)

        plt.figure(self.fig.number)

    def on_epoch_end(self, epoch, logs=None):
        super(PlotLearning, self).on_epoch_end(epoch, logs)
        self.append(epoch, logs)

        if len(self.x) > 0:
            if self.val_output_file is not None:
                data_table = [tuple([self.run_name, self.x[i]] + [v[i] for v in self.values.values()]) for i in
                              range(len(self.x))]
                data_arr = np.asarray(data_table, dtype=[("run", "<U128"), ("epoch", int)] +
                                                        [(k, float) for k in self.values.keys()])
                np.savetxt(self.val_output_file, data_arr, fmt=['%s', '%d'] + ['%f'] * len(list(self.values.keys())),
                           delimiter=",", header="run,epoch," + ",".join(list(self.values.keys())), encoding="utf8")

            if self.show_progress or self.plot_output_file is not None:
                self.replot()

            if self.plot_output_file is not None:
                self.fig.savefig(self.plot_output_file, dpi=100)

            if self.show_progress:
                plt.show(block=False)
                plt.draw()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            tf.logging.warning("Something seems to be wrong with class 'PlotLearning' during training!")

    def on_train_end(self, logs=None):
        if len(self.x) > 0 and self.show_progress:
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
    ref_dict = getImgAndCommandList(os.path.join(args.data_dir, args.ref_dir), onlyUseSubfolder="left_rect",
                                    filterZeros=True, useDiscretCmds=args.useDiscretCmds)

    folders = sorted(list(set([os.path.basename(os.path.split(r["folderPath"])[-2]) for r in ref_dict])))

    print("Select one: ")
    for i, subfolder in enumerate(folders):
        print("{}) {}".format(i + 1, subfolder))

    selection = input()
    filter = folders[int(selection.strip()) - 1]
    json_file = os.path.join(args.run_dir, args.session_dir, "predictions.json")
    ref_vals = [r["velYaw"] for r in ref_dict if filter in r["folderPath"]]
    ref_basenames = [r["fileName"] + r["fileExt"] for r in ref_dict if filter in r["folderPath"]]
    pred_vals = None

    if os.path.exists(json_file):
        with open(json_file) as f:
            predictions = json.load(f)

        # Do some checks before merging the reference and prediction values
        basenames = [p["fileName"] + p["fileExt"] for p in predictions if filter in p["relFolderPath"]]
        pred_vals = [p['predVelYaw'] for p in predictions if filter in p["relFolderPath"]]
        assert len(pred_vals) == len(ref_vals), "Predictions and ground truth have to be of same length!"
        assert all(b == os.path.basename(f) for b, f in zip(basenames, ref_basenames))

    plot_ref_pred_comparison(ref_vals, pred_vals, filter=filter)


def plot_control_balance(args):
    ibg = ImageBatchGenerator(args.data_dir, multi_dir=args.val_dirs, shuffle=False, batch_size=1, crop=False)

    fig, ax = plt.subplots(1, 1)
    ax.title("")
    ax.ylabel("Anzahl")
    ax.xlabel("Drehgeschwindigkeit [%]")
    ax.hist(ibg.labels, bins=[-0.5, -0.001, 0.001, 0.5])

    lower = np.less_equal(ibg.labels, 0.001)
    higher = np.greater_equal(ibg.labels, -0.001)
    between = lower & higher

    print("Nr. of samples between {} and {}: {}".format(-0.001, 0.001, np.sum(between)))

    plt.show()


if __name__ == '__main__':
    plot_parser = argparse.ArgumentParser("Plot the learning curve etc. for trained networks")
    plot_parser.add_argument("--method", action="store", type=str, default="comparison")
    plot_parser.add_argument("--data_dir", action="store", type=str, default=os.path.join(os.path.expanduser("~"),
                                                                                          "volksbot/data"))
    plot_parser.add_argument("--run_dir", action="store", type=str, default=os.path.join(os.path.expanduser("~"),
                                                                                         "volksbot/run"))
    plot_parser.add_argument("--session_dir", action="store", type=str, default="mobilenet_9cls_v6")
    plot_parser.add_argument("--ref_dir", action="store", type=str, default="test_course_oldcfg")
    plot_parser.add_argument("--run", action="append", type=str, default=[])
    plot_parser.add_argument("--val_dir", action="store", type=str, default="test_course_oldcfg")
    plot_parser.add_argument("--val_dirs", action="append", type=str, default=[])
    plot_parser.add_argument("--show_plot", action="store", type=int, default=1)
    plot_parser.add_argument("--output_file", action="store", type=str, default="learning_curves")
    plot_parser.add_argument("--useDiscretCmds", action="store", type=bool, default=False)
    args = plot_parser.parse_args()

    if args.method == "comparison":
        print(args.useDiscretCmds)
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
    elif args.method == "balance":
        plot_control_balance(args)
    else:
        raise NotImplementedError("The method '{}' is not implemented".format(args.method))
