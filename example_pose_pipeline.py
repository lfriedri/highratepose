import argparse
from time import sleep, monotonic
import numpy as np
import os


parser = argparse.ArgumentParser('example_pose_pipeline.py',
                                 description='Movenet inference on live camera input data.')
parser.add_argument('model_path', help='provide complete path to a ...edgetpu.tflite model file')

args = parser.parse_args()

modelPath = args.model_path



from pose_pipeline import PosePipeline
from movenet_process import KEYPOINT_CODES


class NoseTracker:
    def __init__(self):
        self._pipeline = PosePipeline(fps=75, modelPath=modelPath,
                                      nMovenet=2, callback=self._callback)
        self._counter = 0
        self._nMeasure = 500
        self._left = False
    
    def _callback(self, frame, pose):
        self._counter += 1
        
        left = pose[KEYPOINT_CODES.NOSE, 1] < 0.5
        if left and not self._left:
            self._left = True
            print('Nose position changed to left half of image')
        if not left and self._left:
            self._left = False
            print('Nose position changed to right half of image')
        
        # simulate some processing time:
        sleep(0.005)
        
        if self._counter >= self._nMeasure:
            times = self._pipeline.getTimes()
            tAvg = (times[-1, 0] - times[0, 0]) / (times.shape[0] - 1)
            fps = 1 / tAvg
            tMax = np.diff(times[:, 0]).max()
            print(f'{self._nMeasure} frames processed with average time {tAvg} (->{fps} fps) and maximum time {tMax}')
            
            self._counter = 0

    def terminate(self):
        self._pipeline.terminate()
    
    def saveTimes(self):
        np.savetxt('times.csv', self._pipeline.getTimes())


noseTracker = NoseTracker()

print('Entering main loop, press Ctrl-C to finish.')
try:
    while True:
        sleep(0.1)
except KeyboardInterrupt:
    print('terminating by user request')

noseTracker.terminate()

print('saving time data to times.csv')
noseTracker.saveTimes()