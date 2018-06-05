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
parser.add_argument("--show_time", action="store", type=bool, default=False)
parser.add_argument("--crop", action="store", type=bool, default=True)


class Image2TwistNode:
    def __init__(self):
        # Init ROS Node such that we can use other rospy functionality afterwards
        rospy.init_node('img2twist_node')

        # Strip away ROS specific arguments and parse the residual arguments
        self.args = parser.parse_args(rospy.myargv(rospy.myargv()[1:]))

        # If the script is run from launch file, the args might be none. Ask the ros param server instead!
        if self.args.model_file is None and self.args.weight_file is None:
            self.args.model_file = rospy.get_param("~model_file")
            self.args.weight_file = rospy.get_param("~weight_file")
            self.args.x_vel = rospy.get_param("~x_vel")
            self.args.args_file = rospy.get_param("~args_file")
            self.args.show_time = rospy.get_param("~show_time")
            self.args.crop = rospy.get_param("~crop")

        # Check if necessary command line arguments are present
        if self.args.model_file is None or self.args.weight_file is None:
            raise ValueError("Name of 'model_file' and absolute path to 'weight_file' are mandatory!")

        # Set and get Tensorflow specific settings
        tf.logging.set_verbosity(tf.logging.INFO)
        rospy.loginfo("Tensorflow version {} loaded!".format(tf.__version__))

        # Initialize the Keras model and helper from model_file and restore the weights
        self.model, self.helper = build_model(self.args.model_file, for_training=False)
        restore_weights(self.model, self.args.weight_file)
        self.model._make_predict_function()
        self.session = tf.keras.backend.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize()

        self.max_nsecs_delay = 40e6

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
        now_nsecs = rospy.Time.now().nsecs
        # rospy.loginfo("{}".format((now_nsecs-img_msg.header.stamp.nsecs)>self.max_nsecs_delay))
        if (now_nsecs - img_msg.header.stamp.nsecs) > self.max_nsecs_delay:
            return

        # TODO Check: Image conversion from msg -> cv2 (BGR) -> np (RGB)
        np_arr = np.fromstring(img_msg.data, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        np_img = np_img[:, :, ::-1]

        # Crop, resize and rescale pixel values from [0, 1] range to [-1, +1] range
        if self.args.crop:
            np_img = np_img[:, 380:1100, :]

        np_img = resize(np_img, (224, 224))
        np_img = (np_img - 0.5) * 2

        if self.args.show_time:
            end = time.time()
            rospy.loginfo("Prep took: {}".format(np.round(end - start, 4)))
            start = end

        # Run prediction for the current image
        with self.session.as_default():
            with self.graph.as_default():
                output = self.model.predict(np.expand_dims(np_img, axis=0))
                prediction = self.helper.postprocess_output(output)

                # Create the Twist message and fill the respective fields
                cmd = Twist()
                cmd.linear.x = self.args.x_vel
                cmd.angular.z = prediction

                # Send the created message to the roscore
                self.pub.publish(cmd)

            # msg = CompressedImage()
            # msg.header.stamp = rospy.Time.now()
            # msg.format = "jpeg" # "bgr8; jpeg compressed bgr8"
            # resized_img = resized_img[:, :, ::-1]

            # cv2.imshow("img", resized_img)
            # cv2.waitKey(0)

            # msg.data = np.array(cv2.imencode('.jpg', resized_img)[1]).tostring()
            # Publish new image
            # self.img_pub.publish(msg)

        if self.args.show_time:
            end = time.time()
            rospy.loginfo("Pred took: {}".format(np.round(end - start, 4)))


if __name__ == '__main__':
    img2twist_node = Image2TwistNode()