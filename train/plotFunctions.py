import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from matplotlib import style
from inputFunctions import getImgAndCommandList


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
            self.x.append(epoch+1)

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


def main():
    #json_file = "/home/florian/Development/tmp/run/mobilenet_reg_v4/predictions.json"

    #with open(json_file) as f:
    #    predictions = json.load(f)

    references = getImgAndCommandList("/home/florian/Development/tmp/data/test_course_oldcfg",
                                      onlyUseSubfolder="left_rect", filterZeros=True)

    # Do some checks before merging the reference and prediction values
    # basenames = [p["fileName"] + p["fileExt"] for p in predictions]
    # pred_vals = [p['predVelYaw'] for p in predictions]
    # assert len(pred_vals) == len(ibg.labels)
    # assert all(b == os.path.basename(f) for b, f in zip(basenames, ibg.features))

    # Copy the reference values into the same
    # for i, p in enumerate(predictions):
    #     p["refVelYaw"] = data[i]["velYaw"]

    # Do a subselection
    #uniq_folders = list(set([p["relFolderPath"] for p in predictions]))
    #filter = uniq_folders[3]

    refs = [r["velYaw"] for r in references]

    #refs = [r["velYaw"] for r in references if filter in r["folderPath"]]
    #preds = [p["predVelYaw"] for p in predictions if p["relFolderPath"] == filter]

    #plot_ref_pred_comparison(refs, preds, filter=filter)
    plot_ref_pred_comparison(refs, None, filter=filter)


if __name__ == '__main__':
    main()
