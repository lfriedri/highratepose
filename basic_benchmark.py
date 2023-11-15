print('imports')

import argparse
import os
import numpy as np
from time import monotonic
import tflite_runtime.interpreter as tflite


parser = argparse.ArgumentParser('benchmark_main.py',
                                 description='Profiles movenet inference on Google Coral USB accelerator.')
parser.add_argument('model_path', nargs='?', default='/home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite')

args = parser.parse_args()

print('loading delegate')

delegate = tflite.load_delegate('libedgetpu.so.1', {})

modelPath = args.model_path

print(f'loading model {modelPath}')
interpreter = tflite.Interpreter(model_path=modelPath,
                                 experimental_delegates=[delegate])

interpreter.allocate_tensors()

inputInfo = interpreter.get_input_details()
shape = inputInfo[0]['shape']

print(f'creating random input data of shape: {shape}')

input = (np.random.rand(*shape) * 255).astype(np.uint8)

interpreter.set_tensor(inputInfo[0]['index'], input)

nWarmup = 10
nMeasure = 100

print(f'running {nWarmup} warmup cycles')
for i in range(nWarmup):
    interpreter.invoke()

print(f'running {nMeasure} measurement cycles')
start = monotonic()
for i in range(nMeasure):
    interpreter.invoke()

print(f'average cycle time: {(monotonic() - start) / nMeasure}')

del interpreter