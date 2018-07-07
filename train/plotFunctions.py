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

from PIL import Image
from keras.callbacks import Callback
from matplotlib import style, gridspec
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
from inputFunctions import ImageBatchGenerator, getImgAndCommandList
import matplotlib.font_manager as fm
import locale

# Set default latex and german plot settings
#locale.setlocale(locale.LC_NUMERIC, "de_DE.utf8")
#matplotlib.rc('font', family='serif')
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['axes.formatter.use_locale'] = True

gray = "#8a8b8a"
light_orange = "#ffe0b5"
color = {"black": "#000000", "green": "#85be48", "orange": "#ffa500", "blue": "#0fa3b1", "pink": "#6b2d5c",
         "red_pink": "#9e0031", "turquoise": "#7afdd6"}
markers = ["o", "^", ">", "<", "v", "s", "+"]
cc = itertools.cycle(color.values())
m = itertools.cycle(markers)
#prop = fm.FontProperties(fname='/home/florian/Downloads/computer-modern/cmunrm.ttf', size=10)
prop = fm.FontProperties(family="Arial", size=10)


def plot_ref_pred_comparison(reference, predictions=None, filter=None, factor=0.005, start_ind=0, end_ind=None):
    #matplotlib.rc('font', family='serif')
    #matplotlib.rc('text', usetex=True)
    #matplotlib.rcParams['axes.formatter.use_locale'] = True
    bins = np.arange(-0.5, 0.6, 0.1) / factor
    #bins = np.asarray([-0.5, -0.375, -0.25, -0.125, -0.001, 0.001, 0.125, 0.25, 0.375, 0.5]) / factor
    yticks = np.asarray([-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5]) / factor
    #yticks = np.asarray([-0.4375, -0.3125, -0.1875, -0.0625, 0, 0.0625, 0.1875, 0.3125, 0.4375]) / factor

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, gridspec_kw={"width_ratios": [3, 1]})

    ax1.set_title("Steuerbefehl Vergleich - Verlauf", fontproperties=prop)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # ax1.set_ylabel("Gierrate [\%] (Klassen-Nr.)", fontproperties=prop)
    ax1.set_ylabel("Gierrate [%]", fontproperties=prop)
    ax1.set_xlabel("Bild-Nummer", fontproperties=prop)
    ax1.grid(color=gray, linestyle='-', linewidth='1')

    # Plot histogram of controls
    ax2.set_title("Histogramm", fontproperties=prop)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel("Anzahl", fontproperties=prop)
    ax2.set_yticks(yticks)

    #vals = ax.get_yticks()
    #vals = [str(int(x * 100)) for x in vals]
    #ax2.set_yticklabels(["{} ({})".format(a, i) for i, a in enumerate(yticks)])

    # ax2.set_ylim([-80, 100])
    ax2.grid(color=gray, linestyle='-', linewidth='1', zorder=0)

    reference = np.asarray(reference) / factor
    reference = reference[start_ind:end_ind]
    dps = len(reference)

    # if filter is not None:
    #     plt.suptitle(filter)

    # Filter datapoints without predictions
    if predictions is not None:
        for k, pred in predictions.items():
            pred = [p if p != None else np.nan for p in pred]
            pred = np.ma.array(pred)
            pred = np.ma.masked_where(pred == np.nan, pred)
            pred = pred / factor
            pred = pred[start_ind:end_ind]
            predictions[k] = pred

    circle = next(m)

    if predictions is not None:
        for k, pred in predictions.items():
            ax1.plot(range(dps), pred, label="Berechnete Steuerbefehle", color="blue", linewidth=1)
            ax2.hist(pred, bins=bins, orientation='horizontal', histtype='step', color="blue", linewidth=2, zorder=3)

            pred_hist, _ = np.histogram(pred, bins)
            ax2.scatter(pred_hist[4], 0.0, marker="x", color="blue", zorder=6)

    ax1.plot(range(dps), reference, label="Aufgezeichnete Steuerbefehle", color="#00740b", linewidth=2)

    ref_hist, _ = np.histogram(reference, bins)
    ax2.scatter(ref_hist[4], 0.0, marker="x", color="#00740b", zorder=6)
    ax2.hist(np.asarray(reference), bins=bins, orientation='horizontal', histtype='step', color="#00740b", linewidth=2,
             zorder=3)

    ax1.legend(fancybox=True, shadow=True, ncol=1)  # loc='lower center') #, bbox_to_anchor=(0.5, 1.5))

    #fig.tight_layout()
    fig.savefig("../documentation/comp_reg.pdf", pad_inches=0.0)
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
            self.ax.plot(self.x, self.values[metric], label=metric, color=color, linewidth=2, linestyle=linestyle)

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
    color = {"green": "#85be48", "orange": "#ffa500", "blue": "#0fa3b1", "pink": "#6b2d5c",
             "red_pink": "#9e0031", "turquoise": "#7afdd6"}
    markers = ["o", "^", ">", "<", "v", "+"]
    cc = itertools.cycle(color.values())
    m = itertools.cycle(markers)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 6), sharex=True)
    ax1.set_title("Fehlerverlauf", fontproperties=prop)
    ax1.set_ylabel("Mean-Absolute-Error", fontproperties=prop)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.set_title("Genauigkeitsverlauf", fontproperties=prop)
    ax2.set_xlabel("Epochen", fontproperties=prop)
    ax2.set_ylabel("Genauigkeit [\%]", fontproperties=prop)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    data_dict = {}

    for row in data_table:
        if row["run"] not in data_dict:
            data_dict[row["run"]] = {"epoch": [], "loss": [], "metric": []}

        data_dict[row["run"]]["epoch"].append(row["epoch"])
        data_dict[row["run"]]["loss"].append(row["val_loss"])
        data_dict[row["run"]]["metric"].append(row["val_metric"])

    for run in data_dict.keys():
        label = run.replace("mobilenet_", "")

        if "cls" in run:
            ax2.plot(data_dict[run]["epoch"], np.asarray(data_dict[run]["metric"])/0.01, label=label.replace("_", "\_"),
                     color=next(cc), linewidth=1, marker=next(m), markersize=3, zorder=3)
        elif "reg" in run:
            ax1.plot(data_dict[run]["epoch"], np.asarray(data_dict[run]["loss"]), label=label.replace("_", "\_"),
                     color=next(cc), linewidth=1, marker=next(m), markersize=3, zorder=3)


    ax1.legend(fancybox=True, shadow=True, ncol=1)
    ax2.legend(fancybox=True, shadow=True, ncol=1)
    ax1.grid(color=gray, linestyle='-', linewidth='1', zorder=0)
    ax2.grid(color=gray, linestyle='-', linewidth='1', zorder=0)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlim([0, 20])
    ax2.set_ylim([0, 40])
    ax1.set_ylim([0.05, 0.40])

    fig.tight_layout()

    if fig_path is not None:
        fig.savefig(fig_path, dpi=100, pad_inches=0.0)

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
                                    filterZeros=True, useDiscretCmds=args.use_discrete_cmds)

    folders = sorted(list(set([os.path.basename(os.path.split(r["folderPath"])[-2]) for r in ref_dict])))

    print("Select one: ")
    for i, subfolder in enumerate(folders):
        print("{}) {}".format(i + 1, subfolder))

    selection = input()
    filter = folders[int(selection.strip()) - 1]
    ref_vals = [r["velYaw"] for r in ref_dict if filter in r["folderPath"]]
    ref_basenames = [r["fileName"] + r["fileExt"] for r in ref_dict if filter in r["folderPath"]]
    pred_vals = {}

    for v in args.val_dirs:
        json_file = os.path.join(args.run_dir, v, "predictions.json")

        if os.path.exists(json_file):
            with open(json_file) as f:
                predictions = json.load(f)

            # Do some checks before merging the reference and prediction values
            basenames = [p["fileName"] + p["fileExt"] for p in predictions if filter in p["relFolderPath"]]
            pred_vals[v] = [p['predVelYaw'] for p in predictions if filter in p["relFolderPath"]]
            assert len(pred_vals[v]) == len(ref_vals), "Predictions and ground truth have to be of same length!"
            assert all(b == os.path.basename(f) for b, f in zip(basenames, ref_basenames))

    plot_ref_pred_comparison(ref_vals, pred_vals, filter=filter)


def plot_control_balance(args):
    ibg_train = ImageBatchGenerator(args.data_dir, multi_dir=args.val_dirs, shuffle=False, batch_size=1, crop=False)
    ibg_test = ImageBatchGenerator(os.path.join(args.data_dir, "test_course_oldcfg"), shuffle=False, batch_size=1,
                                   crop=False)

    factor = 0.005
    bins = np.asarray([-0.5, -0.375, -0.25, -0.125, -0.001, 0.001, 0.125, 0.25, 0.375, 0.5]) / factor
    yticks = np.asarray([-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5]) / factor
    _ = next(cc)
    _ = next(cc)
    c1 = next(cc)
    c2 = next(cc)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title("Steuerbefehl-Verteilung", fontproperties=prop)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Anzahl", fontproperties=prop)
    ax.set_ylabel("Gierrate [\%]", fontproperties=prop)
    ax.grid(color=gray, linestyle='-', linewidth='1', zorder=0)
    ax.set_yticks(yticks)
    ax.hist(np.asarray(ibg_train.labels) / factor, label="Training", bins=bins, orientation='horizontal',
            histtype="step", linewidth=2, color=c1, zorder=3)
    ax.hist(np.asarray(ibg_test.labels) / factor, label="Test", bins=bins, orientation='horizontal', histtype="step",
            linewidth=2, color=c2, zorder=3)

    counts_train, _ = np.histogram(np.asarray(ibg_train.labels) / factor, bins=bins)
    counts_test, _ = np.histogram(np.asarray(ibg_test.labels) / factor, bins=bins)
    print(np.sum(counts_train))
    print(np.sum(counts_test))

    ax.scatter(counts_train[4], 0, marker="x", color=c1, zorder=6)
    ax.scatter(counts_test[4], 0, marker="x", color=c2, zorder=6)
    ax.legend(fancybox=True, shadow=True, ncol=1)  # loc='lower center') #, bbox_to_anchor=(0.5, 1.5))
    plt.show()

    fig.tight_layout()
    fig.savefig("../documentation/balance.pdf", pad_inches=0.0)
    plt.show()


if __name__ == '__main__':
    plot_parser = argparse.ArgumentParser("Plot the learning curve etc. for trained networks")
    plot_parser.add_argument("--method", action="store", type=str, default="comparison")
    plot_parser.add_argument("--data_dir", action="store", type=str, default=os.path.join(os.path.expanduser("~"),
                                                                                          "volksbot/data"))
    plot_parser.add_argument("--run_dir", action="store", type=str, default=os.path.join(os.path.expanduser("~"),
                                                                                         "volksbot/run"))
    #plot_parser.add_argument("--session_dir", action="store", type=str, default="mobilenet_reg_higher_lr")
    plot_parser.add_argument("--ref_dir", action="store", type=str, default="test_course_oldcfg")
    plot_parser.add_argument("--run", action="append", type=str, default=[])
    plot_parser.add_argument("--val_dir", action="store", type=str, default="test_course_oldcfg")
    plot_parser.add_argument("--val_dirs", action="append", type=str, default=["mobilenet_cls_no_pretrain_c5"])
    plot_parser.add_argument("--show_plot", action="store", type=int, default=1)
    plot_parser.add_argument("--output_file", action="store", type=str, default="learning_curves")
    plot_parser.add_argument("--use_discrete_cmds", action="store", type=int, default=1)
    args = plot_parser.parse_args()

    if args.method == "comparison":
        prepare_comparison_plot(args)
    elif args.method == "learning_curve":
        csv_file_reg = os.path.join(args.run_dir, "learning_curves_reg.csv")
        csv_file_cls = os.path.join(args.run_dir, "learning_curves_cls.csv")

        if os.path.exists(csv_file_reg) and os.path.exists(csv_file_cls):
            data_arr_reg = np.genfromtxt(csv_file_reg, delimiter=",", encoding="utf8", skip_header=1,
                                         dtype=[("run", "<U128"), ("epoch", np.int), ("loss", np.float32),
                                                ("metric", np.float32), ("val_loss", np.float32),
                                                ("val_metric", np.float32)])
            data_arr_cls = np.genfromtxt(csv_file_cls, delimiter=",", encoding="utf8", skip_header=1,
                                         dtype=[("run", "<U128"), ("epoch", np.int), ("loss", np.float32),
                                                ("metric", np.float32), ("val_loss", np.float32),
                                                ("val_metric", np.float32)])
            data_arr = np.concatenate([data_arr_cls, data_arr_reg], axis=0)

            plot_learning_curve(data_arr, fig_path=os.path.join(args.run_dir, "learning_curves.pdf"))
        else:
            prepare_learning_curve_plot(args)
    elif args.method == "balance":
        plot_control_balance(args)
    elif args.method == "quad":
        fig = plt.figure(figsize=(4.5, 4.5))

        plt.subplot(221)
        plt.title("Vortrainiert", fontproperties=prop)
        img = Image.open("../documentation/so_reg_hlr.jpg")
        plt.imshow(np.uint8(img), cmap="jet")
        plt.ylabel("Regression")

        plt.subplot(222)
        plt.title('Zuf{\\"a}llig initialisiert', fontproperties=prop)
        img = Image.open("../documentation/so_reg_nphlr.jpg")
        plt.imshow(np.uint8(img), cmap="jet")

        plt.subplot(223)
        img = Image.open("../documentation/so_cls_hlr.jpg")
        plt.imshow(np.uint8(img), cmap="jet")
        plt.ylabel("Klassifikation", fontproperties=prop)

        plt.subplot(224)
        img = Image.open("../documentation/so_cls_nphlr.jpg")
        plt.imshow(np.uint8(img), cmap="jet")

        fig.tight_layout()
        fig.savefig("../documentation/sa_quad.pdf", pad_inches=0.0)
        plt.show()
    else:
        raise NotImplementedError("The method '{}' is not implemented".format(args.method))
