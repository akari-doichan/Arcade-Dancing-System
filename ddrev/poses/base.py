# coding: utf-8
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BasePoseEstimator(ABC):
    """Abstract Class for Pose Estimation."""

    def __init__(self):
        pass

    @abstractmethod
    def process_frame(
        self, frame: npt.NDArray[np.uint8], key: int = 1, **kwargs
    ) -> npt.NDArray[np.uint8]:
        """[summary]

        Args:
            frame (npt.NDArray[np.uint8]) : A three channel ``BGR`` image represented as numpy ndarray.
            key (int, optional)           :  An integer representing the Unicode character. Defaults to ``1``.

        Returns:
            npt.NDArray[np.uint8] : An edited (drawn ``landmarks``) frame.
        """
        return frame
