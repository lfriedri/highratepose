import numpy as np
import tflite_runtime.interpreter as tflite
from multiprocessing import Process, Event, Array
import ctypes


def _movenetTask(initFinished: Event, newData: Event, calculationFinished: Event,
                 sharedFrame: Array, sharedKeypoints: Array,
                 deviceId: int, modelPath: str):
    initFinished.set()


class MovenetProcess:
    def __init__(self, modelPath: str, deviceId: int):
        print(f'Creating MovenetProcess from {modelPath} on deviceId {deviceId}')
        
        # create temporary interpreter, only to find input shape:
        interpreter = tflite.Interpreter(model_path=modelPath)
        inputInfo = interpreter.get_input_details()
        self._frameShape = inputInfo[0]['shape'][1:]
        print(f'frameShape: {self._frameShape}')

        self._initFinished = Event()
        self._newData = Event()
        self._calculationFinished = Event()
        
        self._sharedFrame = Array(ctypes.c_uint8, int(np.prod(self._frameShape)))
        
        keypointShape = [17, 3] # row, column, confidence -> 3
        self._sharedKeypoints = Array(ctypes.c_float, int(np.prod(keypointShape)))

        
        self._frame = np.frombuffer(self._sharedFrame.get_obj(),
                                    dtype=np.uint8).reshape(self._frameShape)
        self._keypoints = np.frombuffer(self._sharedKeypoints.get_obj(),
                                        dtype=np.dtype('float32')).reshape(keypointShape)

        self._process = Process(target=_movenetTask,
                                args=(self._initFinished,
                                      self._newData,
                                      self._calculationFinished,
                                      self._sharedFrame,
                                      self._sharedKeypoints,
                                      modelPath,
                                      deviceId))
        
        self._process.start()

        self._initFinished.wait()
    
    def findPose(self, frame: np.array) -> None:
        pass
    
    def getPose(self) -> np.array:
        pass
    
    def terminate(self) -> None:
        pass


if __name__ == '__main__':
    mp = MovenetProcess('/home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite', 0)