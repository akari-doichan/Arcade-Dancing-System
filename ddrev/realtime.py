# coding: utf-8
import warnings
from typing import Callable, List, Optional

import cv2
import numpy as np
import numpy.typing as npt

from .recorder import copyVideoSpec
from .utils._colorings import toGREEN, toRED


class VideoCapture(cv2.VideoCapture):
    """Wrapper class for ``cv2.VideoCapture``.

    Args:
        out_path (Optional[str], optional) : The path to which the ``cv2.VideoWriter`` writes video. Defaults to ``None``.
        codec (str, optional)              : Video Codec. Defaults to ``"MP4V"``.
    """

    def __init__(
        self, *args, out_path: Optional[str] = None, codec: str = "MP4V", **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not self.isOpened():
            warnings.warn(
                toRED(
                    "VideoCapture is not opened. Please make sure the device number is correct."
                )
            )
        if out_path is not None:
            self.set_VideoWriter(out_path=out_path, codec=codec)
        else:
            self.out, self.out_path = (None, None)

    def set_VideoWriter(self, out_path: str, codec: str = "MP4V") -> None:
        """Set a ``cv2.VideoWriter`` using :meth:`copyVideoSpec <ddrev.recorder.copyVideoSpec>`

        Args:
            out_path (str)        : The path to which the ``cv2.VideoWriter`` writes video.
            codec (str, optional) : Video Codec. Defaults to ``"MP4V"``.
        """
        _, self.out, self.out_path = copyVideoSpec(
            cap=self, codec=codec, out_path=out_path
        )

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
            if self.out is not None:
                self.out.write(frame)
            if key != -1:
                print(f"{toGREEN(chr(key))} was keyed in.")
                if key in stop_keys:
                    break
        self.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyWindow(winname)
