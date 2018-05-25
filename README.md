# Autonomous driving with neural networks in ROS

This ReadMe explains the concept, that was developed within the "Robotic and Computational Intelligence" project seminar at Technische Universität Darmstadt (Germany). A dynamic environment allows the training and usage of several network models for end2end control of the Frauenhofer IAIS Volksbot RT6 that was equiped with a Stereolabs ZED camera for computer vision.

Detailed information can be found here:

- [Project seminar](http://www.rmr.tu-darmstadt.de/lehre_rmr/vorlesungen_rmr/sommersemester/ps_robotik_ci/index.de.jsp)
- [Volksbot RT6](http://www.volksbot.de/rt6-de.php)
- [Stereolabs ZED camera](https://www.stereolabs.com/zed/)
- [ROS Kinetic Kame](https://wiki.ros.org/kinetic/Installation/Ubuntu)

## Setup / Getting started

Follow the steps below to setup a similar environment that we used during the development phase. If you want to use the `autonomous_driving` package for a different robot you can skip all steps related to the `volksbot-driver` and replace it with !

### Operating System and ROS

As operating system Ubuntu 16.04.4 LTS (Xenial Xerus) with ROS Kinetic Kame is used since it is stated as prerequisite for `zed-ros-wrapper` ([source](https://github.com/stereolabs/zed-ros-wrapper)) by Stereolabs. Besides this, the Ubuntu and ROS versions are in general very well supported and compatible to most software packages.

- For the installation of ROS Kinetic Kame on Ubuntu Xenial Xerus follow the official instruction:

   https://wiki.ros.org/kinetic/Installation/Ubuntu

- After that, create a new ROS workspace with `catkin_make` as shown in the following tutorial:

   https://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment

### ROS Packages

Clone our workspace into the newly created workspace:

```
cd <path-to-your-workspace>/src
git clone https://github.com/robbyHuelsi/VolksbotLineDetection.git ./autonomous_driving
```

In addition to our package you might need one package for retrieving images from a camera (i.e. in our case the Stereolabs ZED camera). If you also have the Stereolabs ZED camera, make sure that the ZED SDK, CUDA and Point Cloud Library are installed before you clone the following repository ([source](https://github.com/stereolabs/zed-ros-wrapper)):

```
cd <path-to-your-workspace>/src
git clone https://github.com/stereolabs/zed-ros-wrapper
```

To control the robot (i.e. in our case the Volksbot RT6) a driver might also be necessary to install. The volksbot driver for example receives velocity commands formated as `geomentry_msgs/Twist` and sets the wheel speed according to its `linear.x` and `angular.z` floating point velocities (more on this later).

```
sudo add-apt-repository ppa:kbs/kbs
sudo apt-get update
sudo apt-get install libepos2-dev

cd <path-to-your-workspace>/src
git clone -b kinetic https://github.com/uos/volksbot_driver
```

If you want to **record your own training data** by controlling the robot via joystick and recording images and control commands at the same time you have to install the following ros packages in addition:

```
sudo apt-get install ros-kinetic-joy
sudo apt-get install ros-kinetic-teleop-twist-joy
```


