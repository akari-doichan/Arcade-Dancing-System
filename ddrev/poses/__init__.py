# coding: utf-8
from typing import List, Union

from ..utils._colorings import toGREEN, toRED
from ..utils._exceptions import KeyError
from .base import BasePoseEstimator

SUPPORTED_MODELS: List[str] = ["mediapipe"]


def get(identifier: Union[str, BasePoseEstimator], **kwargs) -> BasePoseEstimator:
    """Get a Pose Estimator using an identifier.

    Args:
        identifier (Union[str, BasePoseEstimator]): An identifier for Pose Estimator.

    Raises:
        KeyError: When your specified model is not supported.

    Returns:
        BasePoseEstimator: An instance of Pose Estimator

    Examples:
        >>> from ddrev import poses
        >>> from ddrev.poses import BasePoseEstimator
        >>> estimator = poses.get("mediapipe")
        >>> isinstance(estimator, BasePoseEstimator)
        True
    """
    if isinstance(identifier, str):
        if identifier not in SUPPORTED_MODELS:
            sm = ", ".join([f"'{toGREEN(m)}'" for m in SUPPORTED_MODELS])
            raise KeyError(
                f"Please specify an identifier of Pose Estimator from [{sm}], got {toRED(identifier)}"
            )
        if identifier == "mediapipe":
            from .mediapipe import mpPoseEstimator

            return mpPoseEstimator(**kwargs)
    else:
        return identifier
