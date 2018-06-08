#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import tensorflow as tf
import argparse
import time
import matplotlib.pyplot as plt

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from autonomous_driving.train.trainTensorFlow import build_model, restore_recent_weights
from skimage.transform import resize
import keras.applications.mobilenet as mobilenet
from keras.models import load_model

parser = argparse.ArgumentParser(description='Run neural network based line/lane following ROS node.')
parser.add_argument("--x_vel", action="store", type=float, default=0.1)
parser.add_argument("--model_file", action="store", type=str, default=None)
parser.add_argument("--restore_file", action="store", type=str, default=None)
parser.add_argument("--args_file", action="store", type=str, default=None)
parser.add_argument("--show_time", action="store", type=bool, default=False)
parser.add_argument("--crop", action="store", type=int, default=0)


class Image2TwistNode:
    def __init__(self):
        # Init ROS Node such that we can use other rospy functionality afterwards
        rospy.init_node('img2twist_node')

        # Strip away ROS specific arguments and parse the residual arguments
        self.args = parser.parse_args(rospy.myargv(rospy.myargv()[1:]))

        # If the script is run from launch file, the args might be none. Ask the ros param server instead!
        if self.args.model_file is None and self.args.restore_file is None:
            self.args.model_file = rospy.get_param("~model_file")
            self.args.restore_file = rospy.get_param("~restore_file")
            self.args.x_vel = rospy.get_param("~x_vel")
            self.args.args_file = rospy.get_param("~args_file")
            self.args.show_time = rospy.get_param("~show_time")
            self.args.crop = rospy.get_param("~crop")

        # Check if necessary command line arguments are present
        if self.args.model_file is None or self.args.restore_file is None:
            raise ValueError("Name of 'model_file' and absolute path to 'restore_file' are mandatory!")

        # Set and get Tensorflow specific settings
        tf.logging.set_verbosity(tf.logging.INFO)
        rospy.loginfo("Tensorflow version {} loaded!".format(tf.__version__))

        # Initialize the Keras model and helper from model_file and restore the weights
	if self.args.model_file.endswith(".hdf5"):
		tf.logging.info("RESTORE WHOLE MODEL")
		self.model = load_model(self.args.model_file, custom_objects={
					'relu6': mobilenet.mobilenet.relu6})
	else:
        	self.model, self.helper = build_model(self.args.model_file, self.args, for_training=False)
        	restore_recent_weights(self.model, "", self.args.restore_file)

        self.model._make_predict_function()
        self.session = tf.keras.backend.get_session()
        #self.graph = tf.get_default_graph()
        #self.graph.finalize()

	#for layer in self.model.layers:
	#	print(layer.name)
	
	#iweights = self.model.get_layer("conv_pw_13_bn").get_weights()
	
	#for w in weights:
	#	print(w[0])
	#	print(w.shape)
	#print(weights.shape)
	#print(weights)
	#input()

        self.max_delay = rospy.Duration.from_sec(0.04)

        # Initialize the ROS subscriber and publisher and go into loop afterwards
        self.sub = rospy.Subscriber('image', CompressedImage, self.img_callback, queue_size=1)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.img_pub = rospy.Publisher('volksbot_image/compressed', CompressedImage)
        rospy.loginfo("Image2Twist predictor is ready!")
        rospy.spin()

    def img_callback(self, img_msg):
        if self.args.show_time:
            start = time.time()

        # Discard images if they are too old
        now = rospy.Time.now()
        # rospy.loginfo("{}".format((now_nsecs-img_msg.header.stamp.nsecs)>self.max_nsecs_delay))
        if (now - img_msg.header.stamp) > self.max_delay:
            return
	else:
	    rospy.loginfo("{}, delayed by {}ns".format(img_msg.header.stamp, now - img_msg.header.stamp))

        # TODO Check: Image conversion from msg -> cv2 (BGR) -> np (RGB)
        np_arr = np.fromstring(img_msg.data, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        np_img = np_img[:, :, ::-1]

        # Crop, resize and rescale pixel values from [0, 1] range to [-1, +1] range
        if self.args.crop:
            np_img = np_img[:, 380:1100, :]

        np_img = resize(np_img, (224, 224))
	np_img = np.multiply(np.subtract(np_img, 0.5), 2.0)
	#print(np.max(np_img))
	#print(np.min(np_img))

        if self.args.show_time:
            end = time.time()
            rospy.loginfo("Prep took: {}".format(np.round(end - start, 4)))
            start = end

        # Run prediction for the current image
        #with self.session.as_default():
        #    with self.graph.as_default():
        output = self.model.predict(np.expand_dims(np_img, axis=0))
        prediction = self.helper.postprocess_output(output)

        # Create the Twist message and fill the respective fields
        cmd = Twist()
        cmd.linear.x = self.args.x_vel
        cmd.angular.z = prediction[0]

	rospy.loginfo("angular.z: {}".format(cmd.angular.z))

        # Send the created message to the roscore
        self.pub.publish(cmd)

        #msg = CompressedImage()
        #msg.header.stamp = rospy.Time.now()
        #msg.format = "jpeg" # "bgr8; jpeg compressed bgr8"
        #np_img = np_img[:, :, ::-1]

	#plt.clf()
	#plt.cla()
        #plt.imshow((np_img/2.0)+0.5)
        #plt.waitforbuttonpress()

        #msg.data = np.array(cv2.imencode('.jpg', np_img)[1]).tostring()
        # Publish new image
        #self.img_pub.publish(msg)

        if self.args.show_time:
            end = time.time()
            rospy.loginfo("Pred took: {}".format(np.round(end - start, 4)))


if __name__ == '__main__':
    img2twist_node = Image2TwistNode()
