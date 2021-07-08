# coding: utf-8
import warnings
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt

from ._colorings import toRED


def calculate_angle(
    a: npt.NDArray[float],
    b: npt.NDArray[float],
    c: npt.NDArray[float],
    unit: str = "degree",
) -> float:
    """Calculate the counterclockwise angle (``0 ~ 360``) from ``a`` through ``b`` to ``c``.

    Args:
        b (npt.NDArray[float])  : A coordinate of a center point.
        a,c (npt.NDArray[float]): Coordinates for Both end points.
        unit (str)              : Angle unit. Defaults to ``"degree"``.

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
        >>> calculate_angle(*coords)
        90.0
        >>> calculate_angle(*coords, unit="radian")
        1.5707963267948966
    """
    vec_a = a - b
    vec_c = c - b
    if len(a) > 2:
        warnings.warn(
            f"Calclulate angle of {a.ndim}-dimension data from 0 to 180 degrees."
        )
        cos = np.inner(vec_a, vec_c) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_c))
        rad = np.arccos(cos)
    else:
        dot = np.inner(vec_a, vec_c)
        det = vec_c[0] * vec_a[1] - vec_a[0] * vec_c[1]  # determinant
        rad = np.arctan2(det, dot)
    if rad < 0:
        rad += 2 * np.pi
    if unit == "degree":
        return np.rad2deg(rad)
    return rad
