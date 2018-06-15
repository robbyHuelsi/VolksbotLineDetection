import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.layers import Input
from models.mobilenet_reg import model_helper

weight_file = "C:\\Development\\volksbot\\autonomerVolksbot\\run\\mobilenet_reg_lane_v9\\weights_06_0.06.hdf5"
input_shape = (224, 224, 3)
input_tensor = Input(input_shape)
origi_model = MobileNet(weights=None, include_top=False, input_shape=input_shape, input_tensor=input_tensor)
our_model = model_helper.build_model(args=None, for_training=False)
our_model.load_weights(weight_file)

for i, layer in enumerate(origi_model.layers):
    print(i, layer)
