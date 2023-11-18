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

I highly recommend to use an OS installation without graphical user interface, so that the system is not slowed down by the many background tasks of a full desktop installlation.

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

Download ...edgetpu.tflite model files for movenet from https://coral.ai/models/pose-estimation/.

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
You have to provide a local ...edgetpu.tflite model file as command line argument. So for example:
```
python basic_benchmark.py /home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite
```
Typical output:
```
loading delegate
loading model /home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite
creating random input data of shape: [  1 192 192   3]
running 10 warmup cycles
running 100 measurement cycles
average cycle time: 0.01479869901000029
```
## Using MovenetProcess
To run the example, you have to provide a local ...edgetpu.tflite model file as command line argument. So for example:
```
python example_movenet_process.py /home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite
```
Typical output:
```
Loading delegates from separate process for initialization
Creating MovenetProcess from /home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite on deviceId 0
frameShape: [192 192   3]
Starting _movenetTask with /home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite on deviceId 0
Initialization finished
Creating MovenetProcess from /home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite on deviceId 1
frameShape: [192 192   3]
Starting _movenetTask with /home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite on deviceId 1
Initialization finished
running 10 warmup cycles
running 500 measurement cycles
average cycle time (two inferences each): 0.01892996749599115
```
If you use the `MovenetProcess` class from your own code, you should call `findPose()` and `getPose()` in alternating fashion. `findPose()` will take very short time, only starting the inference in the other process. `getPose()` will block until the inference is finished and then return the result. So you can use the time in between for other tasks.

## Using PosePipeline
To run the example, you have to provide a local ...edgetpu.tflite model file as command line argument. So for example:
```
python example_pose_pipeline.py /home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite
```

If you use the `PosePipeline` class from your own code, you can provide a custom `callback` function, that will be called for every new frame and receives the current frame data and the detected human pose keypoints as arguments. Note that you should not reference the frame and the pose data longer than during the callback. If you need to do so, you should make a copy, because they are only memory views.
