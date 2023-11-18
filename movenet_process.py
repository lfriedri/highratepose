import numpy as np
import tflite_runtime.interpreter as tflite
from multiprocessing import Process, Event, Array, set_start_method
import ctypes
import psutil
from cv2 import resize, INTER_NEAREST
import subprocess
import signal
import os


print('Loading delegates from separate process for initialization')

code = """import tflite_runtime.interpreter as tflite
for deviceId in [0, 1]:
    tflite.load_delegate('libedgetpu.so.1', {'device' : f'usb:{deviceId}'})"""

subprocess.run(['python', '-c', code])
subprocess.run(['python', '-c', code])


class KEYPOINT_CODES:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

NUMBER_OF_KEYPOINTS = 17

# row, column, confidence for each keypoint:
KEYPOINT_SHAPE = [NUMBER_OF_KEYPOINTS, 3]


def _movenetTask(initFinished: Event, newData: Event, calculationFinished: Event,
                 sharedFrame: Array, sharedKeypoints: Array,
                 deviceId: int, modelPath: str) -> None:
    """Worker method to be run in other process.
    
    Args:
      initFinished: Is set by _movenetTask when first inference can be started.
      newData: Should be set from outside to signal new input data
      calculationFinished: Is set by _movenetTask after each finished inference
      sharedFrame: shared memory for input data
      sharedKeypoints: shared memory for output data
      deviceId: Google Coral USB accelerator id
      modelPath: ...edgetpu.tflite model file path
    """
    print(f'Starting _movenetTask with {modelPath} on deviceId {deviceId}')
    
    # ignore keyboard interrupts, they are handled by the main application
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    pid = os.getpid()
    os.system(f'sudo chrt -r --pid 50 {pid}')
    
    osProcess = psutil.Process()
    osProcess.cpu_affinity([2 + deviceId])
    
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
    """Run Movenet inference on Google Coral in separate process.
    
    This class sets up a new process and allows to send input data (frames) to it,
    trigger movenet inference and retrieve the output data (keypoints).
    Inference is done using the Google Coral USB accelerator.
    """
    def __init__(self, modelPath: str, deviceId: int) -> None:
        """Constructor.
        
        Args:
          modelPath: ...edgetpu.tflite model file path
          deviceId: Google Coral USB accelerator id
        """
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
        """Trigger inference with given input data.
        
        Args:
          frame: shape: (nx * ny * 3), uint8
            Image will be reshaped to match the input size of the given model.
        """
        resize(frame, tuple(self._frameShape[:2]), self._frame, interpolation=INTER_NEAREST)

        self._newData.set()
    
    def getPose(self) -> np.array:
        """Wait for inference and retrieve output data.
        
        Returns
          nKeypoints * 3 (row, column, confidence)
        """
        self._calculationFinished.wait()
        self._calculationFinished.clear()

        return self._keypoints
    
    def terminate(self) -> None:
        """Terminate the other process for safe shutdown."""
        self._process.terminate()
