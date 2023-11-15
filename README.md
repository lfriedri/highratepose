# highratepose
Human pose estimation with high frame rate on affordable hardware

This repository contains code to run the [Google movenet human pose estimation AI model](https://www.tensorflow.org/hub/tutorials/movenet) on the [Google Coral USB accelerator](https://coral.ai/) in three different usecases:

 - Basic benchmark for inference time using the Google Coral USB accelerator (see basic_benchmark.py)
 - A class `MovenetProcess` (see movenet_process.py) that allows to run movenet inference on the Google Coral USB accelerator in a separate process. Running in a separate process helps to leverage the power of multi core CPUs.\
 There is an example in example_movenet_process.py that runs inferences on two Google Coral USB accelerators in parallel.
 - A class `PosePipeline` (see pose_pipeline.py) that provides a gstreamer pipeline that reads live camera images from a V4L2 camera, performs human pose estimation on the camera images (using two Google Coral USB accelerators in parallel) and runs a custom callback function that can work with the estimated pose data.\
 The example in example_pose_pipeline.py tracks the lateral coordinate of the nose and prints every time it changes between left and right half of the camera image.

The project is described in more detail here: [highratepose hackaday project](http://www.hackaday.io)

# Setup

The following describes the setup on a Raspberry Pi 4.

## OS

It is not straightforward to use the Google Coral USB accelerator with python > 3.9. Therefor, install "Raspberry Pi OS (Legacy) Lite" (Bullseye from 2023-05-03) with Raspberry Pi Imager v1.8.1 to have python 3.9 available.

## Prerequisites for running the basic benchmark

Install the libraries required for Google Coral USB accelerator:
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update

sudo apt-get install libedgetpu1-max

sudo apt-get install python3-tflite-runtime
```

## Additional prerequisites for using MovenetProcess
Install opencv for image resizing:
```
sudo apt-get install python3-opencv
```
<!--sudo apt-get install python3-psutil-->

## Additional prerequisites for using PosePipeline
Install gstreamer for camera input:
```
sudo apt-get install python3-gst-1.0 gstreamer1.0-plugins-good gstreamer1.0-tools libgstreamer1.0-dev
```

# How to run

## Running the basic benchmark

## Using MoventProcess

## Using PosePipeline