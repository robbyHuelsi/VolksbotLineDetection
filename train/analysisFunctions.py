import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
from keras import Model
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.engine import Layer
from keras.layers import Input, Deconv2D
from keras.optimizers import Adam
from tabulate import tabulate
from scipy.ndimage.interpolation import zoom

from inputFunctions import ImageBatchGenerator
from trainTensorFlow import build_model


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


def get_layers(model, filter=None):
    result = [{"id": i, "layer": layer, "name": layer.name} for i, layer in enumerate(model.layers)
              if filter is None or filter in layer.name]

    return result


class MeanLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(MeanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MeanLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return K.mean(x, axis=self.axis, keepdims=True)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = 1
        return tuple(output_shape)


class UpsamplingLayer(Layer):
    def __init__(self, **kwargs):
        super(UpsamplingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UpsamplingLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return K.resize_images(x, 2, 2, data_format="channels_last")

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = output_shape[1] * 2
        output_shape[2] = output_shape[2] * 2
        return tuple(output_shape)


class MultLayer(Layer):
    def __init__(self, **kwargs):
        super(MultLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MultLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return x * kwargs["y"]

    def compute_output_shape(self, input_shape):
        return input_shape


def build_salient_model(our_model):
    activation_layers = get_layers(our_model, "relu")

    for layer_info in activation_layers:
        layer_info["mean_actv"] = MeanLayer(3)(layer_info["layer"].output)

    for i in reversed(range(len(activation_layers) - 1)):
        previous = activation_layers[i + 1]["mean_actv"]
        current = activation_layers[i]["mean_actv"]

        if current.get_shape()[1] != previous.get_shape()[1]:
            previous = UpsamplingLayer()(previous)

        activation_layers[i]["mean_actv"] = MultLayer()(previous, y=current)

    last = UpsamplingLayer()(activation_layers[0]["mean_actv"])
    model = Model(inputs=our_model.layers[0].output, outputs=last)

    return model


if __name__ == '__main__':
    analysis_parser = argparse.ArgumentParser("Analyse model, weights and difference between models.")
    analysis_parser.add_argument("--method", action="store", type=str, default="diff")
    analysis_parser.add_argument("--ref_model", action="store", type=str, default="mobilenet")
    analysis_parser.add_argument("--ref_model_weights", action="store", type=str, default="imagenet")
    analysis_parser.add_argument("--our_model", action="store", type=str, default="mobilenet_reg")
    analysis_parser.add_argument("--our_model_weights", action="store", type=str, default=None)
    analysis_parser.add_argument("--data_dir", action="store", type=str, default=None)
    analysis_parser.add_argument("--img_file", action="store", type=str, default=None)
    args = analysis_parser.parse_args()

    assert args.data_dir

    ref_model = None
    our_model = None

    if args.ref_model == "mobilenet":
        input_shape = (224, 224, 3)
        input_tensor = Input(input_shape)
        ref_model = MobileNet(weights=args.ref_model_weights, include_top=False, input_shape=input_shape,
                              input_tensor=input_tensor)
    else:
        assert args.ref_model_weights

        ref_model, helper = build_model(args.ref_model, args=None, for_training=False)
        ref_model.load_weights(args.ref_model_weights)

    if args.our_model:
        our_model, helper = build_model(args.our_model, args=None, for_training=False)

        if args.our_model_weights:
            print("Restore weights from '{}' for our model!".format(args.our_model_weights))
            our_model.load_weights(args.our_model_weights)
        else:
            print("Don't restore any weights for our model!")

    if args.method == "diff":
        print("Difference between:")
        print("- {} with weights {}".format(args.ref_model, args.ref_model_weights))
        print("- {} with weights {}".format(args.our_model, args.our_model_weights))
        data_table = layer_wise_abs_diff(ref_model, our_model)
        print(tabulate(data_table, headers=["Layer Name", "Kernel Shape", "Avg", "Std", "Sum"]))
    elif args.method == "activations":
        print("Select one layer by its number:")
        show_all_layer_names(our_model, filter="relu")
        print("Select one layer by its number:")
        number = int(input())
        layer_name = our_model.layers[number].name
        print("Your selection: {}".format(layer_name))

        ibg = ImageBatchGenerator(args.data_dir, preprocess_input=preprocess_input, labeled=False, shuffle=False,
                                  crop=False, batch_size=1)
        lam = layer_activation_model(our_model, layer_name=layer_name)

        for img in ibg:
            # actv_fn=lambda x: relu6(x)/6.0)
            plot_actv(lam.predict(img), img=(img[0, :, :, :] + 1.0) / 2.0, actv_fn=lambda x: x / 6.0)
            plt.waitforbuttonpress()
    elif args.method == "salient":
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        model = build_salient_model(our_model)

        if args.img_file:
            img = Image.open(args.img_file)
            img = img.resize((224, 224))
            np_img = preprocess_input(np.float32(img))
            np_img = np.expand_dims(np_img, axis=0)
            pred = model.predict(np_img)
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

            plt.subplot(121)
            plt.title("Image")
            plt.imshow(np.float32(img)/255.0)
            plt.subplot(122)
            plt.title("Salient Map")
            plt.imshow(pred[0, :, :, 0], cmap="plasma")
            plt.show()
        else:
            ibg = ImageBatchGenerator(args.data_dir, preprocess_input=preprocess_input, labeled=False, shuffle=False,
                                      crop=False, batch_size=1)
            preds = model.predict_generator(ibg, verbose=1)

            for i, img in enumerate(ibg):
                if not i % 25 == 0:
                    continue

                pred = preds[i, :, :, :]
                pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
                img = (img[0, :, :, :] + 1.0) / 2.0
                img[:, :, 1] += pred[:, :, 0]
                img[:, :, 1] = np.clip(img[:, :, 1], 0.0, 1.0)

                plt.subplot(121)
                plt.title("Image")
                plt.imshow(img)
                plt.subplot(122)
                plt.title("Salient Map")
                plt.imshow(pred[:, :, 0], cmap="plasma")
                plt.show(block=False)
                plt.waitforbuttonpress()
    else:
        raise NotImplementedError("The method '{}' is not implemented".format(args.method))
