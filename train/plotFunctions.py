import json
import numpy as np
import matplotlib.pyplot as plt

from inputFunctions import getImgAndCommandList


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

    plt.subplot(212)
    both = np.concatenate([np.asmatrix(reference), np.asmatrix(predictions)], axis=0).transpose()
    plt.hist(both, bins=11, orientation='vertical', histtype='bar', color=['xkcd:orange', 'xkcd:sky blue'], label=["Reference", "Prediction"])
    plt.legend()

    plt.show()


def main():
    json_file = "/home/florian/Development/tmp/run/mobilenet_reg_v4/predictions.json"

    with open(json_file) as f:
        predictions = json.load(f)

    references = getImgAndCommandList("/home/florian/Development/tmp/data/train_lane",
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
    uniq_folders = list(set([p["relFolderPath"] for p in predictions]))
    filter = uniq_folders[3]

    refs = [r["velYaw"] for r in references if filter in r["folderPath"]]
    preds = [p["predVelYaw"] for p in predictions if p["relFolderPath"] == filter]

    plot_ref_pred_comparison(refs, preds, filter=filter)


if __name__ == '__main__':
    main()
