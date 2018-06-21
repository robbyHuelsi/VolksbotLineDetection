import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Input
from keras.optimizers import Adam
from tabulate import tabulate

from inputFunctions import ImageBatchGenerator
from models.mobilenet_reg import model_helper


def layer_wise_abs_diff(orig_model, our_model):
    data_table = []

    for i, layer in enumerate(orig_model.layers):
        if "input" in layer.name or "pad" in layer.name or "bn" in layer.name or "relu" in layer.name:
            continue

        # print("{}: {} vs. {}".format(i, layer.name, our_model.layers[i].name))
        w_orig = layer.get_weights()
        w_our = our_model.layers[i].get_weights()

        if len(w_orig) > 0:
            w_orig = w_orig[0]
            w_our = w_our[0]

        abs_diff = np.absolute(w_orig - w_our)
        data_table.append([layer.name, w_orig.shape, np.mean(abs_diff), np.std(abs_diff), np.sum(abs_diff)])

    return data_table


def layer_activation_model(model, layer_name="conv1_relu"):
    layer = [l for l in model.layers if l.name == layer_name]

    assert len(layer) == 1, "Found {} layers with name '{}' instead of 1!".format(len(layer), layer_name)

    model = Model(inputs=model.layers[0].output, outputs=layer[0].output)
    model.compile(Adam(), loss="mae", metrics=["mse"])
    model._make_predict_function()

    return model


def plot_actv(actv, img=None, actv_fn=None, block=False):
    assert actv.shape[0] == 1

    s = actv.shape
    h, w = s[1], s[2]
    m = np.ceil(np.sqrt(s[3]))
    ns = (int(m * s[1]), int(m * s[2]))
    fms = np.ones(ns) * 0.5

    plt.cla()
    plt.clf()

    for ind in range(s[3]):
        col = ind % m
        row = ind // m
        rs, re = int(row * h), int(row + 1) * h
        cs, ce = int(col * w), int(col + 1) * w

        fm = actv[:, :, :, ind]

        if actv_fn is not None:
            fm = actv_fn(fm)

        fms[rs:re, cs:ce] = fm

    if img is not None:
        plt.subplot(121)
        plt.title("Image {}".format(np.random.randn()))
        plt.imshow(img)
        plt.subplot(122)
        plt.title("Feature Maps")

    plt.imshow(fms, cmap="magma")
    plt.show(block=block)


def kernel_statistics(model, layer_names=[]):
    data_table = []

    for layer_name in layer_names:
        for w in model.get_layer(layer_name).get_weights():
            data_table.append([layer_name, w.shape, np.mean(w), np.std(w)])

    return data_table, ["Layer", "Shape", "Mean", "Std"]


def relu6(np_arr):
    return np.minimum(np.maximum(np_arr, np.zeros_like(np_arr)), np.ones_like(np_arr) * 6.0)


def show_all_layer_names(model, filter=None):
    for i, layer in enumerate(model.layers):
        if filter is None or filter in layer.name:
            print("{}: {}, {}".format(i, layer.name, layer.output.shape))


if __name__ == '__main__':
    weight_file = "C:\\Development\\volksbot\\autonomerVolksbot\\run\\mobilenet_reg_lane_v9\\weights_06_0.06.hdf5"
    input_shape = (224, 224, 3)
    input_tensor = Input(input_shape)
    orig_model = MobileNet(weights=None, include_top=False, input_shape=input_shape, input_tensor=input_tensor)
    our_model = model_helper.build_model(args=None, for_training=False)
    our_model.load_weights(weight_file)

    # TASK 1
    # data_table = layer_wise_abs_diff(orig_model, our_model)
    # print(tabulate(data_table, headers=["Layer Name", "Kernel Shape", "Avg", "Std", "Sum"]))

    # TASK 2
    print("All activation layer names:")
    show_all_layer_names(our_model, filter="relu")

    ibg = ImageBatchGenerator("C:\\Development\\volksbot\\autonomerVolksbot\\data\\test_course\\",
                              preprocess_input=preprocess_input, labeled=False, shuffle=False, crop=False, batch_size=1)
    lam = layer_activation_model(our_model, layer_name="conv_dw_2_relu")

    for img in ibg:
        plot_actv(lam.predict(img), img=(img[0, :, :, :] + 1.0) / 2.0, actv_fn=lambda x: x/6.0)  # , actv_fn=lambda x: relu6(x)/6.0)
        plt.waitforbuttonpress()
