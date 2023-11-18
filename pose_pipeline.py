import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gst, GstVideo
import numpy as np
from collections import deque
from typing import Callable
from time import monotonic

from movenet_process import MovenetProcess


Gst.init(None)

Gst.debug_set_active(True)
Gst.debug_set_default_threshold(3)


class PosePipeline:
    """Acquire live images and provide pose detection results.
    
    This class creates a gstreamer pipeline to read image data from
    /dev/video0 at the given frame rate. Each frame is cropped to a
    quadratic format and then given to one of two MovenetProcesses,
    that perform human pose detection, providing keypoint coordinates.
    
    For each frame, the user defined callback function is called with
    the current frame and the detected keypoints as arguments. Note that
    you should not reference the frame and the pose data longer than
    during the callback. If you need to do so, you should make a copy,
    because they are only memory views.
    """
    def __init__(self, fps: int, modelPath: str, nMovenet: int,
                 callback: Callable[[np.array, np.array], None]) -> None:
        """Constructor.
        
        Args:
          fps: frame rate (1 / seconds)
          modelPath: ...edgetpu.tflite model file path
          nMovenet: number of MovenetProcesses to use
          callback: user defined callback function, that accepts numpy arrays
            for frame and pose as arguments
        """
        pipelineDescription = f'v4l2src device=/dev/video0 ! video/x-raw,width=320,height=240,framerate={fps}/1'
        pipelineDescription += ' ! videocrop left=40 right=40' # 240 x 240
        # pipelineDescription += ' ! queue max-size-buffers=1'
        pipelineDescription += ' ! videoconvert ! video/x-raw,format=RGB'
        pipelineDescription += ' ! appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false'
        
        self._pipeline = Gst.parse_launch(pipelineDescription)

        self._sink = self._pipeline.get_by_name('sink')
        self._sink.connect('new-sample', self._onNewFrame)
        
        self._movenets = deque([])
        input = (np.random.rand(*[240, 240, 3]) * 255).astype(np.uint8)
        for i in range(nMovenet):
            movenet = MovenetProcess(modelPath, i)
            movenet.findPose(input)
            self._movenets.append(movenet)
        
        self._callback = callback
        
        self._times = deque([], maxlen=500)
            
        self._pipeline.set_state(Gst.State.PLAYING)
                
    def _onNewFrame(self, sink):
        """Event handler for new frame events."""
        t0 = monotonic()
        sample = sink.emit('pull-sample')

        buffer = sample.get_buffer()
        success, mapInfo = buffer.map(Gst.MapFlags.READ)
        frame = np.ndarray(shape=(240, 240, 3),
                          dtype=np.uint8,
                          buffer=mapInfo.data)
        
        # give the new frame to the first movenet in the deque
        # and put it to the end of the deque
        movenet = self._movenets.popleft()
        movenet.findPose(frame)
        self._movenets.append(movenet)
        
        t1 = monotonic()
        
        # wait for the result of the next movenet in the deque
        pose = self._movenets[0].getPose()
        
        t2 = monotonic()
        
        self._callback(frame, pose)
        
        t3 = monotonic()
        
        self._times.append([t0, t1, t2, t3])
        
        buffer.unmap(mapInfo)
        return Gst.FlowReturn.OK

    def terminate(self) -> None:
        """Terminate the pipeline and all MovenetProcesses."""
        self._pipeline.set_state(Gst.State.NULL)
                
        for movenet in self._movenets:
            movenet.terminate()
    
    def getTimes(self) -> np.array:
        """Get timing info.
        
        Returns timing info for the latest 500 new frame events.
        Each entry has four results from time.monotonic():
        
        1) at the beginning of the new frame handler
        2) after the new frame was sent to worker process
        3) after receiving keypoint data from the other
           worker process
        4) after custom callback was processed
        
        Returns: shape 500 * 4, dtype: float
          In case it is called too early,
          fist dimension may be smaller.
        """
        return np.array(self._times)
        

