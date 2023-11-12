# highratepose
Human pose estimation with high frame rate on affordable hardware

This repository contains code to run the Google movenet human pose estimation AI model in three different usecases:

 - Basic benchmark for inference time using the Google Coral USB accelerator ('benchmark_main.py')
 - A class (PosePipeline in 'pose_pipeline.py') that provides a gstreamer pipeline that reads live camera images from a V4L2 camera, performs human pose estimation on the camera images and runs a custom callback function that can work with the estimated pose data
 - An example application that uses PosePipeline

The project is described in more detail here: [highratepose hackaday project](http://www.hackaday.io)

# Setup

The following describes the setup on a Raspberry Pi 4.

## OS

Install "Raspberry Pi OS (Legacy) Lite (Bullseye from 2023-05-03) with Raspberry Pi Imager v1.8.1

## Prerequisites for running the basic benchmark

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update

sudo apt-get install libedgetpu1-max
(enable maximum operating frequency)

sudo apt-get install python3-tflite-runtime

## Prerequisites for using PosePipeline

sudo apt-get install python3-opencv
sudo apt-get install python3-psutil


**You will need all prerequisites given for the basic benchmark**

# How to run

## Running the basic benchmark

## Running the example application

## Using PosePipeline from your own application