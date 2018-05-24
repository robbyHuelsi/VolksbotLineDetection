#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import tensorflow as tf
import argparse
import time

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from autonomous_driving.train.trainTensorFlow import build_model, restore_weights
from skimage.transform import resize


parser = argparse.ArgumentParser(description='Run neural network based line/lane following ROS node.')
parser.add_argument("--x_vel", action="store", type=float, default=0.1)
parser.add_argument("--model_file", action="store", type=str, default=None)
parser.add_argument("--weight_file", action="store", type=str, default=None)
parser.add_argument("--args_file", action="store", type=str, default=None)


class ANNLineFollower():
    def __init__(self):
        # Init ROS Node such that we can use other rospy functionality afterwards
        rospy.init_node('line_follower')
        self.args = parser.parse_args(rospy.myargv(rospy.myargv()[1:]))

        # Check for necessary command line arguments
        if self.args.model_file is None or self.args.weight_file is None:
            raise ValueError("Absolute paths to 'model_file' and 'weight_file' are mandatory!")

        # Set tensorflow specific settings
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info("Tensorflow version {} loaded!".format(tf.__version__))

        # Initialize the keras model and helper from model_file and restore the weights
        self.model, self.helper = build_model(self.args.model_file, for_training=False)
        restore_weights(self.model, self.args.weight_file)
        self.model._make_predict_function()
        self.session = tf.keras.backend.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize()
        
        # Initialize the ROS subscriber and publisher and go into loop afterwards
        self.sub = rospy.Subscriber('image', CompressedImage, self.img_callback, queue_size=1)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.loginfo("LineFollower is ready!")

        rospy.spin()

    def img_callback(self, img_msg):
        #start = time.time()
        
        # TODO Check: Image conversion from msg -> cv2 (BGR) -> np (RGB)
        np_arr = np.fromstring(img_msg.data, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        np_img = np_img[:, :, ::-1]

        # Convert from [0, 255] range to [-1, +1] range
        np_img = ((np_img / 255.0) - 0.5) * 2
	cropped_img = np_img[380:1100, ]
	resized_img = resize(cropped_img, (224, 224)) #, anti_aliasing=True)

        #end = time.time()
        #rospy.loginfo("Prep took: {}".format(np.round(end-start, 4)))
        #start = end
        
        with self.session.as_default():
            with self.graph.as_default():
                output = self.model.predict(np.expand_dims(resized_img, axis=0))
                prediction = self.helper.postprocess_output(output)

                # Create the Twist message and fill the respective fields
                cmd = Twist()
                cmd.linear.x = self.args.x_vel
                cmd.angular.z = prediction

                # Send the created message to the roscore
                self.pub.publish(cmd)

        #end = time.time()
        #rospy.loginfo("Pred took: {}".format(np.round(end-start, 4)))


if __name__ == '__main__':
    follower = ANNLineFollower()
