#!/usr/bin/env python

import rospy
import os
import csv
import time
import numpy as np
import cv2
import datetime
import sys

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist

home_dir = os.path.expanduser("~")
full_dir = os.path.join(home_dir, 'recordings')
runtime = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")[:-7]
name = sys.argv[1]

if name is not None:
    runtime = name
    print(runtime)

img_dir = os.path.join(full_dir, runtime)
img_dir_left = os.path.join(full_dir, runtime, "left")
img_dir_right = os.path.join(full_dir, runtime, "right")
img_dir_left_rect = os.path.join(full_dir, runtime, "left_rect")
img_dir_right_rect = os.path.join(full_dir, runtime, "right_rect")
# img_rate = 0.03
# last_stamp = 0
cv_bridge = CvBridge()


def img_callback(img_msg, subdir="", file_type="jpg"):
    global img_dir, last_stamp, img_rate

    # Do not always save an image
    # if time.time() - last_stamp > img_rate:
    #    last_stamp = time.time()

    if file_type == "png":
        cv2_img = cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv2.imwrite(os.path.join(img_dir, subdir, '%s.png' % img_msg.header.stamp), cv2_img)
    elif file_type == "jpg":
        np_arr = np.fromstring(img_msg.data, np.uint8)
        encoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(img_dir, subdir, '%s.jpg' % img_msg.header.stamp), encoded_img)


def cmd_callback(cmd_msg):
    global full_dir, runtime

    with open(os.path.join(full_dir, 'cmd_vel_%s.csv' % runtime), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([time.time(),
                         cmd_msg.linear.x, cmd_msg.linear.y, cmd_msg.linear.z,
                         cmd_msg.angular.x, cmd_msg.angular.y, cmd_msg.angular.z])


def recorder():
    # Create the output directory if it does not exist
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir_left)
        os.makedirs(img_dir_right)
        os.makedirs(img_dir_left_rect)
        os.makedirs(img_dir_right_rect)

    # Initialize the node and subscribe to two topics
    rospy.init_node('recorder', anonymous=True)

    # Subscribe to the correct topics (that the camera is publishing) here
    rospy.Subscriber('/left/image_rect_color/compressed', CompressedImage, lambda x: img_callback(x, "left_rect"))
    rospy.Subscriber('/right/image_rect_color/compressed', CompressedImage, lambda x: img_callback(x, "right_rect"))
    rospy.Subscriber('/cmd_vel', Twist, cmd_callback)

    rospy.loginfo("Recorder ready, started at %s." % runtime)

    # Spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    recorder()
