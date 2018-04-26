#!/usr/bin/env python
    
import rospy
import os
import sys
import time
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
#from cv_bridge import CvBridge

import matplotlib.pyplot as plt


class ANNLineFollower():
    def __init__(self):
        # Initialize Tensorflow session
	self.sess = tf.Session()
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info("TF Version %s loaded!" % str(tf.__version__))

        # TODO Replace the following lines with checkpoint loading
        image = tf.placeholder(tf.float32, [1, 480, 640, 3], 'image')
        net = slim.conv2d(image, 10, [3, 3], stride=2, scope='conv1')
        net = layers.flatten(net)
        net = slim.fully_connected(net, 2, activation_fn=tf.tanh)
        linear, angular = tf.split(net, 2, axis=1)
        linear = tf.identity(linear, name='linear')
        angular = tf.identity(angular, name='angular')
        self.sess.run(tf.global_variables_initializer())

	# Get output tensors by name
        # TODO Adopt to the names according to the loaded checkpoint. How to do it?
        self.linear = self.sess.graph.get_tensor_by_name('linear:0')
        self.angular = self.sess.graph.get_tensor_by_name('angular:0')

        # Initialize the ROS subscriber and publisher and go into loop afterwards
	rospy.init_node('line_follower')
        self.sub = rospy.Subscriber('image', CompressedImage, self.img_callback, queue_size=1)
	self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.loginfo("LineFollower is ready!")

        self.once = True

        rospy.spin()

    def img_callback(self, img_msg):
        # TODO Check: Image conversion from msg -> cv2 (BGR) -> np (RGB)
        np_arr = np.fromstring(img_msg.data, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        np_img = np_img[:, :, ::-1]

        # Convert from [0, 255] range to [-1, +1] range
        if self.once:
            plt.imshow(np_img)
            plt.show()
            self.once =False

        np_img = ((np_img / 255.0) - 0.5) * 2

        # Do the prediction of the linear and angular values
        linear_pred, angular_pred = self.sess.run([self.linear, self.angular], feed_dict={'image:0': np_img[np.newaxis, :]})

        # TODO Depending on the activation function in the last layer we might need to clip those values!

        # Create the Twist message and fill the respective fields
	cmd = Twist()
	cmd.linear.x = linear_pred
	cmd.angular.z = angular_pred

        # Send the created message to the roscore
	self.pub.publish(cmd)


if __name__ == '__main__':
    follower = ANNLineFollower()

