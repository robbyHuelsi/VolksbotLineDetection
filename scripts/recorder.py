#!/usr/bin/env python

import rospy
import os
# import sys
import csv
import time
import numpy as np
import cv2
import datetime

# from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

home_dir = os.path.expanduser("~")
full_dir = os.path.join(home_dir, 'recordings')
runtime = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")[:-7]
img_dir = os.path.join(full_dir, runtime)
img_rate = 0.03
last_stamp = 0


def img_callback(img_msg):
    global img_dir, last_stamp, img_rate

    # Do not always save an image
    if time.time() - last_stamp > img_rate:
        last_stamp = time.time()

        np_arr = np.fromstring(img_msg.data, np.uint8)
        encoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(img_dir, '%s.jpg' % img_msg.header.stamp), encoded_img)


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
        os.makedirs(img_dir)

    # Initialize the node and subscribe to two topics
    rospy.init_node('recorder', anonymous=True)
    rospy.Subscriber('/image', CompressedImage, img_callback)
    rospy.Subscriber('/cmd_vel', Twist, cmd_callback)

    rospy.loginfo("Recorder ready, started at %s." % runtime)

    # Spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    recorder()
