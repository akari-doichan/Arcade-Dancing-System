# coding: utf-8
"""Pose Estimation using ``mediapipe``.

- GitHub: https://github.com/google/mediapipe
- Documentation: https://google.github.io/mediapipe/

The landmark model in MediaPipe Pose predicts the location of 33 pose landmarks (see figure from documentation below).

.. image:: https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from ..utils._exceptions import KeyError
from ..utils.feedback_utils import drawScoreArc, putScoreText
from ..utils.score_utils import calculate_angle
from .base import BasePoseEstimator

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

VISIBILITY_THRESHOLD = 0.5


class mpPoseEstimator(mp_pose.Pose, BasePoseEstimator):
    """MediaPipe Pose Estimation Model

    Args:
        static_image_mode (bool, optional)                 : Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream. Defaults to ``False``.
        model_complexity (int, optional)                   : Complexity of the pose landmark model: ``0``, ``1`` or ``2``. Defaults to ``1``.
        smooth_landmarks (bool, optional)                  : Whether to filter landmarks across different input images to reduce jitter. Defaults to ``True``.
        min_detection_confidence (float, optional)         : Minimum confidence value ``([0.0, 1.0])`` for person detection to be considered successful. Defaults to ``0.5``.
        min_tracking_confidence (float, optional)          : Minimum confidence value ``([0.0, 1.0])`` for the pose landmarks to be considered tracked successfully. Defaults to ``0.5``.
        landmark_drawing_specs (Dict[str,Any], optional)   : Keyword Arguments for :meth:`set_landmark_drawing_spec <ddrev.poses.mediapipe.mpPoseEstimator.set_landmark_drawing_spec>`. Defaults to ``{}``.
        connection_drawing_specs (Dict[str,Any], optional) : Keyword Arguments for :meth:`set_connection_drawing_spec <ddrev.poses.mediapipe.mpPoseEstimator.set_connection_drawing_spec>`. Defaults to ``{}``.

    Attributes:
        landmark_drawing_spec (mp_drawing.DrawingSpec)   : Drawing spec for landmarks.
        connection_drawing_spec (mp_drawing.DrawingSpec) : Drawing spec for connections.
    """

    ANGLE_POINTS: List[List[int]] = [
        [18, 16, 14],
        [12, 14, 16],
        [14, 12, 24],
        [12, 24, 26],
        [28, 26, 24],
        [26, 28, 32],
        [13, 15, 17],
        [15, 13, 11],
        [23, 11, 13],
        [25, 23, 11],
        [23, 25, 27],
        [31, 27, 25],
    ]

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
        """Initialize an instance of ``mp.solutions.poses.Pose``.

        Args:
            static_image_mode (bool, optional)                 : Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream. Defaults to ``False``.
            model_complexity (int, optional)                   : Complexity of the pose landmark model: ``0``, ``1`` or ``2``. Defaults to ``1``.
            smooth_landmarks (bool, optional)                  : Whether to filter landmarks across different input images to reduce jitter. Defaults to ``True``.
            min_detection_confidence (float, optional)         : Minimum confidence value ``([0.0, 1.0])`` for person detection to be considered successful. Defaults to ``0.5``.
            min_tracking_confidence (float, optional)          : Minimum confidence value ``([0.0, 1.0])`` for the pose landmarks to be considered tracked successfully. Defaults to ``0.5``.
            landmark_drawing_specs (Dict[str,Any], optional)   : Keyword Arguments for :meth:`set_landmark_drawing_spec <ddrev.poses.mediapipe.mpPoseEstimator.set_landmark_drawing_spec>`. Defaults to ``{}``.
            connection_drawing_specs (Dict[str,Any], optional) : Keyword Arguments for :meth:`set_connection_drawing_spec <ddrev.poses.mediapipe.mpPoseEstimator.set_connection_drawing_spec>`. Defaults to ``{}``.
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
        """Set a drawing spec for landmarks.

        Args:
            color (Tuple[int, int, int], optional) : Color for drawing the annotation. Defaults to red color; ``(0, 0, 255)``.
            thickness (int, optional)              : Thickness for drawing the annotation. Defaults to ``2``.
            circle_radius (int, optional)          : Circle radius. Defaults to ``2``.

        Examples:
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> estimator = mpPoseEstimator()
            >>> estimator.set_landmark_drawing_spec(color=(236,163,245), thickness=10, circle_radius=10)
        """
        self.landmark_drawing_spec = mp_drawing.DrawingSpec(
            color=color, thickness=thickness, circle_radius=circle_radius
        )

    def set_connection_drawing_spec(
        self,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        circle_radius: int = 2,
    ) -> None:
        """Set a drawing spec for landmarks.

        Args:
            color (Tuple[int, int, int], optional) : Color for drawing the annotation. Defaults to green color; ``(0, 255, 0)``.
            thickness (int, optional)              : Thickness for drawing the annotation. Defaults to ``2``.
            circle_radius (int, optional)          : Circle radius. Defaults to ``2``.

        Examples:
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> estimator = mpPoseEstimator()
            >>> estimator.set_connection_drawing_spec(color=(236,163,245), thickness=10, circle_radius=2)
        """
        self.connection_drawing_spec = mp_drawing.DrawingSpec(
            color=color, thickness=thickness, circle_radius=circle_radius
        )

    def process(self, frame: npt.NDArray[np.uint8]) -> NormalizedLandmarkList:
        """Do Pose-Estimation.

        Args:
            frame (npt.NDArray[np.uint8]): A three channel ``BGR`` image represented as numpy ndarray.

        Returns:
            NormalizedLandmarkList: Pose Estimation Landmarks.

        Examples:
            >>> import cv2
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> image = cv2.imread("sample.jpg") # BGR images
            >>> estimator = mpPoseEstimator()
            >>> landmarks = estimator.process(image)
        """
        return super().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks

    def process_frame(
        self,
        frame: npt.NDArray[np.uint8],
        key: int = -1,
        inplace: bool = True,
        **kwargs
    ) -> npt.NDArray[np.uint8]:
        """Process frame in while-loop in :meth:`realtime_process <ddrev.realtime.VideoCapture.realtime_process>`.

        Args:
            frame (npt.NDArray[np.uint8]) : A three channel ``BGR`` image represented as numpy ndarray.
            key (int, optional)           : An integer representing the Unicode character. Defaults to ``1``.
            inplace (bool, optional)      : Whether frame is edited (drawn ``landmarks``) in place. Defaults to ``True``.

        Returns:
            npt.NDArray[np.uint8]: An edited (drawn ``landmarks``) frame.

        .. plot::
         :class: popup-img

            >>> import cv2
            >>> import matplotlib.pyplot as plt
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> from ddrev.utils._path import SAMPLE_IMAGE
            >>> image = cv2.imread(SAMPLE_IMAGE) # BGR images
            >>> estimator = mpPoseEstimator()
            >>> image_edited = estimator.process_frame(image, inplace=False)
            >>> fig, axes = plt.subplots(ncols=2, figsize=(12,4))
            >>> for ax,img,title in zip(axes, [image, image_edited], ["Original", "Pose-Estimated"]):
            ...     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ...     ax.axis("off")
            ...     ax.set_title(title, fontsize=18)
            >>> fig.show()
        """
        landmarks = self.process(frame)
        frame = self.draw_landmarks(frame, landmarks=landmarks, inplace=inplace)
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

        .. plot::
          :class: popup-img

            >>> import cv2
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> from ddrev.utils._path import SAMPLE_IMAGE
            >>> image = cv2.imread(SAMPLE_IMAGE) # BGR images
            >>> estimator = mpPoseEstimator()
            >>> landmarks = estimator.process(image)
            >>> image_edited = estimator.draw_landmarks(image, landmarks, inplace=False)
            >>> fig, axes = plt.subplots(ncols=2, figsize=(12,4))
            >>> for ax,img,title in zip(axes, [image, image_edited], ["Original", "Pose-Estimated"]):
            ...     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ...     ax.axis("off")
            ...     ax.set_title(title, fontsize=18)
            >>> fig.show()
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
        """Convert from landmarks (``NormalizedLandmarkList``) to a string (``str``).

        Args:
            landmarks (NormalizedLandmarkList) : Landmarks.
            encoding (str, optional)           : The encoding with which to decode the bytes. Defaults to ``"latin-1"``.

        Returns:
            str: A string.

        Examples:
            >>> import cv2
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> image = cv2.imread("sample.jpg") # BGR images
            >>> estimator = mpPoseEstimator()
            >>> landmarks = estimator.process(image)
            >>> landmarks_str = estimator.landmarks2string(landmarks)
            >>> isinstance(landmarks_str, str)
            True
        """
        return landmarks.SerializeToString().decode(encoding)

    @staticmethod
    def string2landmarks(
        string: str, encoding: str = "latin-1"
    ) -> NormalizedLandmarkList:
        """Convert from a string (``str``) to landmarks (``NormalizedLandmarkList``).

        Args:
            string (str)             : A string.
            encoding (str, optional) : The encoding in which to encode the string. Defaults to ``"latin-1"``.

        Returns:
            NormalizedLandmarkList: Landmarks.

        Examples:
            >>> import cv2
            >>> import matplotlib.pyplot as plt
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> with open("landmarks.txt") as f:
            ...     landmarks_str = f.read()
            >>> image = cv2.imread("sample.jpg") # BGR images
            >>> estimator = mpPoseEstimator()
            >>> landmarks = estimator.string2landmarks(landmarks_str)
            >>> image_edited = estimator.draw_landmarks(image, landmarks, inplace=False)
            >>> fig, axes = plt.subplots(ncols=2, figsize=(12,4))
            >>> for img,ax in zip([image, image_edited], axes):
            ...     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            >>> fig.show()
        """
        return NormalizedLandmarkList.FromString(string.encode(encoding))

    @staticmethod
    def calculate_angle(
        landmarks: NormalizedLandmarkList,
        angle_points: List[List[int]] = ANGLE_POINTS,
        unit: bool = "radian",
    ) -> npt.NDArray[float]:
        """Calculate angles of each ``angle_points``.

        Args:
            landmarks (NormalizedLandmarkList)                 : Landmarks.
            angle_points (Optional[List[List[int]]], optional) : A list of 3 points used to determine the angle. Defaults to ``ANGLE_POINTS``.
            unit (str, optional)                               : Unit of Angle. Defaults to ``"radian"``.

        Returns:
            npt.NDArray[float]: A list of angles.

        Examples:
            >>> import cv2
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> from ddrev.utils._path import SAMPLE_IMAGE
            >>> img = cv2.imread(SAMPLE_IMAGE)
            >>> estimator = mpPoseEstimator()
            >>> landmarks = estimator.process(img)
            >>> angles = estimator.calculate_angle(landmarks)
            >>> len(angles)
            12
        """
        angles = [-1] * len(angle_points)
        if (landmarks is None) or (not hasattr(landmarks, "landmark")):
            return angles
        landmark = landmarks.landmark
        for i, points in enumerate(angle_points):
            coords = []
            for point in points:
                p = landmark[point]
                if p.visibility < VISIBILITY_THRESHOLD:
                    break
                # coords.append(np.asarray([p.x, p.y, p.z]))
                coords.append(np.asarray([p.x, p.y]))
            if len(coords) == 3:
                angles[i] = calculate_angle(*coords, unit=unit)
        return np.asarray(angles)

    @staticmethod
    def draw_score(
        frame: npt.NDArray[np.uint8],
        scores: List[float],
        landmarks: NormalizedLandmarkList,
        angle_points: List[List[int]] = ANGLE_POINTS,
        draw_func: Callable[
            [npt.NDArray[np.uint8], List[List[float]], float, bool],
            npt.NDArray[np.uint8],
        ] = drawScoreArc,
        inplace: bool = True,
        **kwargs
    ) -> npt.NDArray[np.uint8]:
        """Draw ``score`` in ``frame`` using ``draw_func``

        Args:
            frame (npt.NDArray[np.uint8])                                                                           : Input image.
            scores (List[float])                                                                                    : Scores to display.
            landmarks (NormalizedLandmarkList)                                                                      : Landmarks.
            angle_points (List[List[int]])                                                                          : A list of 3 points used to determine the angle. Defaults to ``ANGLE_POINTS``.
            draw_func (Callable[ [npt.NDArray[np.uint8], List[List[float]], float, bool], npt.NDArray[np.uint8], ]) : How to draw the ``score``. Defaults to :meth:`drawScoreArc <ddrev.utils.feedback_utils.drawScoreArc>`.
            inplace (bool, optional)                                                                                : Whether frame is edited (drawn ``score``) in place. Defaults to ``:meth:drawScoreArc``.

        Returns:
            npt.NDArray[np.uint8]: An edited (drawn ``score`` using ``draw_func``) image.

        .. plot::
         :class: popup-img

            >>> import cv2
            >>> import matplotlib.pyplot as plt
            >>> from ddrev.poses.mediapipe import mpPoseEstimator
            >>> from ddrev.utils import drawScoreArc, putScoreText
            >>> from ddrev.utils._path import SAMPLE_IMAGE
            >>> fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
            >>> image = cv2.imread(SAMPLE_IMAGE)  # BGR images
            >>> estimator = mpPoseEstimator()
            >>> landmarks = estimator.process(image)
            >>> estimator.draw_landmarks(image, landmarks, inplace=True)
            >>> scores = [(i+1)/len(landmarks.landmark) for i in range(len(landmarks.landmark))]
            >>> for draw_func, ax in zip([drawScoreArc, putScoreText], axes):
            ...     img = estimator.draw_score(
            ...         frame=image, scores=scores, landmarks=landmarks,
            ...         draw_func=draw_func, inplace=False
            ...     )
            ...     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ...     ax.axis("off")
            ...     ax.set_title(draw_func.__name__, fontsize=18)
            >>> fig.show()
        """
        if (landmarks is None) or (not hasattr(landmarks, "landmark")):
            return frame
        landmark = landmarks.landmark
        for i, (points, score) in enumerate(zip(angle_points, scores)):
            coords = []
            for point in points:
                p = landmark[point]
                if p.visibility < VISIBILITY_THRESHOLD:
                    break
                # coords.append(np.asarray([p.x, p.y, p.z]))
                coords.append(np.asarray([p.x, p.y]))
            if len(coords) == 3:
                frame = draw_func(
                    frame=frame, score=score, coords=coords, inplace=inplace, **kwargs
                )
        return frame
