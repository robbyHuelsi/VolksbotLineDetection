import json
import matplotlib.pyplot as plt

from inputFunctions import ImageBatchGenerator


def main():
    ibg = ImageBatchGenerator("/home/florian/Development/tmp/data/train_lane", shuffle=False, batch_size=1)

    #plt.figure()
    #plt.hist(ibg.labels, bins=11)
    #plt.show()

    json_file = "/home/florian/Development/tmp/run/mobilenet_reg_v3/predictions.json"

    with open(json_file) as f:
        predictions = json.load(f)

    pred_vals = [p['predVelYaw'] for p in predictions]
    print(ibg.features[220])


    assert len(pred_vals) == len(ibg.labels)

    plt.figure()
    plt.plot(range(len(ibg.labels)), ibg.labels, label="Soll")
    plt.plot(range(len(ibg.labels)), pred_vals, label="Ist")
    # plt.fill_between(range(len(ibg.labels)), ibg.labels, pred_vals)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
