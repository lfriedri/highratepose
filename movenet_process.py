import numpy as np
import tflite_runtime.interpreter as tflite
from multiprocessing import Process, Event, Array, set_start_method
import ctypes
import psutil
from cv2 import resize, INTER_NEAREST
import subprocess


print('Loading delegates from separate process for initialization')

code = """import tflite_runtime.interpreter as tflite
for deviceId in [0, 1]:
    tflite.load_delegate('libedgetpu.so.1', {'device' : f'usb:{deviceId}'})"""

subprocess.run(['python', '-c', code])
subprocess.run(['python', '-c', code])


# 17 keypoints
# each row, column, confidence
KEYPOINT_SHAPE = [17, 3]


def _movenetTask(initFinished: Event, newData: Event, calculationFinished: Event,
                 sharedFrame: Array, sharedKeypoints: Array,
                 deviceId: int, modelPath: str):
    print(f'Starting _movenetTask with {modelPath} on deviceId {deviceId}')
    
    osProcess = psutil.Process()
    osProcess.cpu_affinity([2, 3])
    
    delegate = tflite.load_delegate('libedgetpu.so.1', {'device' : f'usb:{deviceId}'})
    
    interpreter = tflite.Interpreter(model_path=modelPath,
                                     experimental_delegates=[delegate])
    
    interpreter.allocate_tensors()
    
    inputInfo = interpreter.get_input_details()
    frameShape = inputInfo[0]['shape'][1:]
    inputIndex = inputInfo[0]['index']
    
    outputInfo = interpreter.get_output_details()
    outputIndex = outputInfo[0]['index']
    
    frame = np.frombuffer(sharedFrame.get_obj(),
                          dtype=np.uint8).reshape(frameShape)
    keypoints = np.frombuffer(sharedKeypoints.get_obj(),
                              dtype=np.dtype('float32')).reshape(KEYPOINT_SHAPE)
    
    print('Initialization finished')
    initFinished.set()

    while True:
        newData.wait()
        newData.clear()
        
        interpreter.tensor(inputIndex)()[0] = frame
        interpreter.invoke()
        
        result = interpreter.tensor(outputIndex)()
        keypoints[:] = result.reshape(KEYPOINT_SHAPE)
        del result
        
        calculationFinished.set()


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
        
        self._sharedKeypoints = Array(ctypes.c_float, int(np.prod(KEYPOINT_SHAPE)))

        
        self._frame = np.frombuffer(self._sharedFrame.get_obj(),
                                    dtype=np.uint8).reshape(self._frameShape)
        self._keypoints = np.frombuffer(self._sharedKeypoints.get_obj(),
                                        dtype=np.dtype('float32')).reshape(KEYPOINT_SHAPE)

        self._process = Process(target=_movenetTask,
                                args=(self._initFinished,
                                      self._newData,
                                      self._calculationFinished,
                                      self._sharedFrame,
                                      self._sharedKeypoints,
                                      deviceId,
                                      modelPath))
        
        self._process.start()

        self._initFinished.wait()
    
    def findPose(self, frame: np.array) -> None:
        resize(frame, tuple(self._frameShape[:2]), self._frame, interpolation=INTER_NEAREST)

        self._newData.set()
    
    def getPose(self) -> np.array:
        self._calculationFinished.wait()
        self._calculationFinished.clear()

        return self._keypoints
    
    def terminate(self) -> None:
        self._process.terminate()


if __name__ == '__main__':
    from time import monotonic
   
    
    modelPath = '/home/pi/models/movenet_single_pose_lightning_ptq_edgetpu.tflite'
    mps = [MovenetProcess(modelPath, i) for i in [0, 1]]        
    
    input = (np.random.rand(*[240, 240, 3]) * 255).astype(np.uint8)
    
    for i in range(10):
        for mp in mps:
            mp.findPose(input)
        for mp in mps:
            mp.getPose()
        
    n = 500
    start = monotonic()
    for i in range(n):
        for mp in mps:
            mp.findPose(input)
        for mp in mps:
            mp.getPose()
    print((monotonic() - start) / (2 * n))
    
    for mp in mps:
        mp.terminate()