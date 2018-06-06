import json
import os
import matplotlib.pyplot as plt
import numpy as np

from inputFunctions import ImageBatchGenerator


def plot_ref_pred_comparison(reference, predictions, filter=None):
    plt.figure()
    plt.title("Control command comparison")

    if filter is not None:
        plt.suptitle(filter)

    plt.subplot(211)
    plt.title("Yaw Velocity")
    plt.plot(range(len(reference)), reference, label="Reference", color='xkcd:orange', linewidth=2)
    plt.plot(range(len(predictions)), predictions, label="Prediction", color='xkcd:sky blue', linewidth=2)
    plt.grid(color='gray', linestyle='-', linewidth='1')
    plt.legend()

    # plt.subplot(212)
    # plt.title("Yaw Integral = Position")
    # plt.plot(range(len(reference)), np.cumsum(reference), label="Reference", color='xkcd:orange', linewidth=2)
    # plt.plot(range(len(predictions)), np.cumsum(predictions), label="Prediction", color='xkcd:sky blue', linewidth=2)
    # plt.grid(color='gray', linestyle='-', linewidth='1')
    # plt.legend()

    plt.subplot(212)
    both = np.concatenate([np.asmatrix(reference), np.asmatrix(predictions)], axis=0).transpose()
    plt.hist(both, bins=11, orientation='vertical', histtype='bar', color=['xkcd:orange', 'xkcd:sky blue'], label=["Reference", "Prediction"])
    plt.legend()

    plt.show()


def main():
    ibg = ImageBatchGenerator("/home/florian/Development/tmp/data/train_lane", shuffle=False, batch_size=1)
    json_file = "/home/florian/Development/tmp/run/mobilenet_reg_v3/predictions.json"

    with open(json_file) as f:
        predictions = json.load(f)

    # Do some checks before merging the reference and prediction values
    basenames = [p["fileName"] + p["fileExt"] for p in predictions]
    pred_vals = [p['predVelYaw'] for p in predictions]
    assert len(pred_vals) == len(ibg.labels)
    assert all(b == os.path.basename(f) for b, f in zip(basenames, ibg.features))

    # Copy the reference values into the same
    for i, p in enumerate(predictions):
        p["refVelYaw"] = ibg.labels[i]

    # Do a subselection
    uniq_folders = list(set([p["relFolderPath"] for p in predictions]))
    filter = uniq_folders[3]

    refs = [p["refVelYaw"] for p in predictions if p["relFolderPath"] == filter]
    preds = [p["predVelYaw"] for p in predictions if p["relFolderPath"] == filter]

    plot_ref_pred_comparison(refs, preds, filter=filter)


if __name__ == '__main__':
    main()
