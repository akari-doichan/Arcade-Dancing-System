# coding: utf-8

from typing import List

import numpy as np
import numpy.typing as npt


def calculate_angle(
    a: npt.NDArray[float],
    b: npt.NDArray[float],
    c: npt.NDArray[float],
    unit: str = "radian",
) -> float:
    """Calculate angle from 3 coordinates.

    Args:
        b (npt.NDArray[float])  : A coordinate of a center point.
        a,c (npt.NDArray[float]): Coordinates for Both end points.
        unit (str)              : Angle unit. Defaults to ``"radian"``.

    Returns:
        float: Angles of ``a``, ``b``, ``c`` around ``b``.

    Examples:
        >>> import numpy as np
        >>> from ddrev.utils import calculate_angle
        >>> coords = [
        ...     np.asarray([0, 1]),
        ...     np.asarray([0, 0]),
        ...     np.asarray([1, 0]),
        >>> ]
        >>> calculate_angle(*coords, unit="degree")
        90.0
        >>> calculate_angle(*coords)
        1.5707963267948966
    """
    vec_a = a - b
    vec_c = c - b
    cos = np.inner(vec_a, vec_c) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_c))
    rad = np.arccos(cos)
    if unit == "degree":
        return np.rad2deg(rad)
    return rad
