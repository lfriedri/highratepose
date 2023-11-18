import argparse
import numpy as np
from time import monotonic


parser = argparse.ArgumentParser('example_movenet_process.py',
                                 description='Use two Google Coral USB accelerators from two separate processes in parallel.')
parser.add_argument('model_path', help='provide complete path to a ...edgetpu.tflite model file')

args = parser.parse_args()

modelPath = args.model_path


from movenet_process import MovenetProcess


movenets = [MovenetProcess(modelPath, i) for i in [0, 1]]        

input = (np.random.rand(*[240, 240, 3]) * 255).astype(np.uint8)

nWarmup = 10
nMeasure = 500

print(f'running {nWarmup} warmup cycles')
for i in range(nWarmup):
    for movenet in movenets:
        movenet.findPose(input)
    for movenet in movenets:
        movenet.getPose()

print(f'running {nMeasure} measurement cycles')
start = monotonic()
for i in range(nMeasure):
    for movenet in movenets:
        movenet.findPose(input)
    for movenet in movenets:
        movenet.getPose()

print(f'average cycle time (two inferences each): {(monotonic() - start) / nMeasure}')

for movenet in movenets:
    movenet.terminate()