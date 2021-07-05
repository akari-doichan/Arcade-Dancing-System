# coding: utf-8
import warnings
from typing import Callable, List

import cv2
import numpy as np
import numpy.typing as npt

from .utils._colorings import toGREEN, toRED


class VideoCapture(cv2.VideoCapture):
    """Wrapper class for ``cv2.VideoCapture``.

    Attributes:
        tm (cv2.TickMeter) :
        max_count (int)    :
    """

    def __init__(self, *args, max_count: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.isOpened():
            warnings.warn(
                toRED(
                    "VideoCapture is not opened. Please make sure the device number is correct."
                )
            )

    #     self.tm = cv2.TickMeter()
    #     self.max_count = max_count

    # def calc_fps(self):
    #     self.tm.stop()
    #     fps = self.max_count / self.tm.getTimeSec()
    #     self.tm.reset()
    #     self.tm.start()

    @classmethod
    def check_device(cls) -> None:
        """Check the connected device number.

        Examples:
            >>> from ddrev.realtime import VideoCapture
            >>> VideoCapture.check_device()
            Device Number 0 is found
            Device Number 1 is found
            Device Number 2 is NOT found
            ==============================
            2 devices are connected.
        """
        idx = 0
        while True:
            cap = cls(idx)
            flag = not cap.isOpened()
            msg = toRED("NOT found") if flag else toGREEN("found")
            print(f"Device Number {toGREEN(idx)} is {msg}")
            cap.release()
            if flag:
                break
            idx += 1
        print(f"{'='*30}\n{toGREEN(idx)} devices are connected.")

    def describe(self) -> None:
        """Describe the device information.

        Examples:
            >>> from ddrev.realtime import VideoCapture
            >>> cap = VideoCapture(0)
            >>> cap.describe()
            [Device Information]
                    Width  : 1280.0
                    Height : 720.0
                    FPS    : 29.000049
            >>> cap.release()
        """
        print(
            f"""[Device Information]
        Width  : {toGREEN(self.get(cv2.CAP_PROP_FRAME_WIDTH))}
        Height : {toGREEN(self.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        FPS    : {toGREEN(self.get(cv2.CAP_PROP_FPS))} """
        )

    @staticmethod
    def _do_nothing(frame: npt.NDArray[np.uint8], key: int):
        """Do nothing."""
        return frame

    def realtime_process(
        self,
        function: Callable[[npt.NDArray[np.uint8], int], npt.NDArray[np.uint8]] = None,
        stop_keys: List[int] = [27, ord("q")],
        delay: int = 1,
        winname: str = "Realtime Demonstration",
    ) -> None:
        """Do realtime video processing.

        Args:
            function (Callable[[npt.NDArray[np.uint8], int], npt.NDArray[np.uint8]], optional): Function to process BGR image from webcame and return BGR image. Defaults to ``self._do_nothing``.
            stop_keys (List[int], optional)                                         : Enter these keys to end the process. Defaults to ``[27, ord("q")]``.
            delay (int, optional)                                                   : Waits for a pressed key [ms]. Defaults to ``1``.
            winname (str, optional)                                                 : The window name that visualizes the results of real-time video processing. Defaults to ``"Realtime Demonstration"``.

        Examples:
            >>> from ddrev.realtime import VideoCapture
            >>> cap = VideoCapture(0)
            >>> cap.realtime_process()
        """
        if (function is None) or (not callable(function)):
            warnings.warn(
                f"No video processing function {toGREEN('function')} was given, so do nothing (using {toGREEN('self._do_nothing')} instead.)"
            )
            function = self._do_nothing
        while True:
            is_ok, frame = self.read()
            if (not is_ok) or (frame is None):
                break
            key = cv2.waitKey(delay=delay)
            frame = function(frame, key)
            cv2.imshow(winname=winname, mat=frame)
            if key in stop_keys:
                break
        self.release()
        cv2.destroyWindow(winname)
