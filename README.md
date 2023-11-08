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

Install "Raspberry Pi OS Lite (64-bit)" (Bookworm from 2023-10-10) with the Raspberry Pi Imager v1.8.1.

## Prerequisites for running the basic benchmark

## Prerequisites for using PosePipeline

# How to run

## Running the basic benchmark

## Running the example application

## Using PosePipeline from your own application