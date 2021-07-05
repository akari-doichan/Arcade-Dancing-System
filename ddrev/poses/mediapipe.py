# coding: utf-8
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from ..utils._exceptions import KeyError
from .base import BasePoseEstimator

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class mpPoseEstimator(mp_pose.Pose, BasePoseEstimator):
    """MediaPipe Pose Estimation Model

    Args:
        static_image_mode (bool, optional) : [description]. Defaults to ``False``.
        model_complexity (int, optional) : [description]. Defaults to ``1``.
        smooth_landmarks (bool, optional) : [description]. Defaults to ``True``.
        min_detection_confidence (float, optional) : [description]. Defaults to ``0.5``.
        min_tracking_confidence (float, optional) : [description]. Defaults to ``0.5``.

    Attributes:
        landmarks (NormalizedLandmarkList) : Pose landmarks as a result of the most recent :meth:`process <ddrev.pose.mediapipe.process>`
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        landmark_drawing_specs: Dict[str, Any] = {},
        connection_drawing_specs: Dict[str, Any] = {},
        **kwargs
    ):
        """Initialize an instance of ``mp.solutions.pose.Pose``.

        Args:
            static_image_mode (bool, optional)                 : Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream. Defaults to ``False``.
            model_complexity (int, optional)                   : Complexity of the pose landmark model: ``0``, ``1`` or ``2``. Defaults to ``1``.
            smooth_landmarks (bool, optional)                  : Whether to filter landmarks across different input images to reduce jitter. Defaults to ``True``.
            min_detection_confidence (float, optional)         : Minimum confidence value ``([0.0, 1.0])`` for person detection to be considered successful. Defaults to ``0.5``.
            min_tracking_confidence (float, optional)          : Minimum confidence value ``([0.0, 1.0])`` for the pose landmarks to be considered tracked successfully. Defaults to ``0.5``.
            landmark_drawing_specs (Dict[str,Any], optional)   : Keyword Arguments for :meth:`set_landmark_drawing_spec <ddrev.pose.mediapipe.mpPoseEstimator.set_landmark_drawing_spec>`. Defaults to ``{}``.
            connection_drawing_specs (Dict[str,Any], optional) : Keyword Arguments for :meth:`set_connection_drawing_spec <ddrev.pose.mediapipe.mpPoseEstimator.set_connection_drawing_spec>`. Defaults to ``{}``.
        """
        super().__init__(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.set_landmark_drawing_spec(**landmark_drawing_specs)
        self.set_connection_drawing_spec(**connection_drawing_specs)

    def set_landmark_drawing_spec(
        self,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        circle_radius: int = 2,
    ) -> None:
        self.landmark_drawing_spec = mp_drawing.DrawingSpec(
            color=color, thickness=thickness, circle_radius=circle_radius
        )

    def set_connection_drawing_spec(
        self,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        circle_radius: int = 2,
    ) -> None:
        self.connection_drawing_spec = mp_drawing.DrawingSpec(
            color=color, thickness=thickness, circle_radius=circle_radius
        )

    def process(self, frame: npt.NDArray[np.uint8]) -> NormalizedLandmarkList:
        """Do Pose-Estimation.

        Args:
            frame (npt.NDArray[np.uint8]): A three channel ``BGR`` image represented as numpy ndarray.

        Returns:
            NormalizedLandmarkList: Pose Estimation Landmarks.
        """
        return super().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks

    def process_frame(
        self, frame: npt.NDArray[np.uint8], key: int = -1, **kwargs
    ) -> npt.NDArray[np.uint8]:
        """Process frame in while-loop in :meth:`realtime_process <ddrev.realtime.VideoCapture.realtime_process>`

        Args:
            frame (npt.NDArray[np.uint8]) : A three channel ``BGR`` image represented as numpy ndarray.
            key (int, optional)           : An integer representing the Unicode character. Defaults to ``1``.

        Returns:
            npt.NDArray[np.uint8]: An edited (drawn ``landmarks``) frame.
        """
        landmarks = self.process(frame)
        self.draw_landmarks(frame, landmarks=landmarks, inplace=True)
        return frame

    def draw_landmarks(
        self,
        frame: npt.NDArray[np.uint8],
        landmarks: NormalizedLandmarkList,
        inplace: bool = True,
    ) -> npt.NDArray[np.uint8]:
        """Draws the landmarks and the connections on the image.

        Args:
            frame (npt.NDArray[np.uint8])      : A three channel ``RGB`` image represented as numpy ndarray.
            landmarks (NormalizedLandmarkList) : A normalized landmark list proto message to be annotated on the image.
            inplace (bool, optional)           : Whether frame is edited (drawn ``landmarks``) in place. Defaults to ``True``.

        Returns:
            npt.NDArray[np.uint8]: An edited (drawn ``landmarks``) frame.
        """
        if not inplace:
            frame = frame.copy()
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.landmark_drawing_spec,
            connection_drawing_spec=self.connection_drawing_spec,
        )
        return frame

    @staticmethod
    def landmarks2string(
        landmarks: NormalizedLandmarkList, encoding: str = "latin-1"
    ) -> str:
        return landmarks.SerializeToString().decode(encoding)

    @staticmethod
    def string2landmarks(
        string: str, encoding: str = "latin-1"
    ) -> NormalizedLandmarkList:
        return NormalizedLandmarkList.FromString(string.encode(encoding))
