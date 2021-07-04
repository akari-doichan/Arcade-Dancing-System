# coding: utf-8
from typing import Any, Dict, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class mpPoseEstimator(mp_pose.Pose):
    """[summary]

    Args:
        static_image_mode (bool, optional) : [description]. Defaults to ``False``.
        model_complexity (int, optional) : [description]. Defaults to ``1``.
        smooth_landmarks (bool, optional) : [description]. Defaults to ``True``.
        min_detection_confidence (float, optional) : [description]. Defaults to ``0.5``.
        min_tracking_confidence (float, optional) : [description]. Defaults to ``0.5``.
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

    def process(self, frame: npt.NDArray[np.uint8], key: Optional[int] = None, inplace: bool = True, draw: bool = True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = super().process(rgb)
        self._landmarks = results.pose_landmarks
        if draw and self._landmarks:
            if not inplace:
                frame = frame.copy()
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=self._landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_drawing_spec,
                connection_drawing_spec=self.connection_drawing_spec,
            )
        return frame

    @property
    def landmarks(self):
        return self._landmarks
