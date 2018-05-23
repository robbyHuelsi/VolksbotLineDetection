#!/bin/bash

python3 trainTensorFlow.py --model_file="mobilenet_reg" --run_dir="/tmp/run/" --session_name="pred" --data_dir="/home/roboterlabor/" --train_dir="recordings_vs" --save_file="weights.19-0.07.hdf5" --epochs=0
