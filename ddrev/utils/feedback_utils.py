# coding: utf-8
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import Colormap

from .score_utils import calculate_angle


def score2color(
    score: float, cmap: Union[str, Colormap] = "coolwarm_r"
) -> Tuple[int, int, int]:
    """Convert score to RGB color.

    Args:
        score (float)                        : Score normalized between ``0`` and ``1``.
        cmap (Union[str,Colormap], optional) : Color map to apply. Defaults to ``"coolwarm_r"``.

    Returns:
        Tuple[int,int,int]: RGB color corresponding to the score.

    Examples:
        >>> from ddrev.utils import score2color
        >>> score2color(score=0.2)
        (123, 158, 248)
        >>> score2color(score=0.5)
        (221, 220, 219)
        >>> score2color(score=1)
        (59, 77, 193)
    """
    cmap = plt.get_cmap(cmap)
    return tuple([int(255 * e) for e in cmap(score)][:3])  # [::-1]


def cmap_indicator_create(
    width: int,
    height: int,
    transpose: bool = False,
    turnover: bool = False,
    cmap: Union[str, Colormap] = "coolwarm_r",
) -> npt.NDArray[np.uint8]:
    """Create a colormap indicator

    Args:
        width (int), height (int)             : The size of the created indicator.
        transpose (bool, optional)            : Whether to transpose it or not. Defaults to ``False``.
        turnover (bool, optional)             : Whether to turn it over or not. Defaults to ``False``.
        cmap (Union[str, Colormap], optional) : Color map to apply. Defaults to ``"coolwarm_r"``.

    Returns:
        npt.NDArray[np.uint8]: Color map indicator.

    .. plot::
      :class: popup-img

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from ddrev.utils import cmap_indicator_create
        >>> fig, axes = plt.subplots(ncols=2,nrows=2,figsize=(12,8))
        >>> for ax_r,transpose in zip(axes, [False, True]):
        ...     for ax,turnover in zip(ax_r, [False, True]):
        ...         ax.imshow(cmap_indicator_create(width=100, height=50, transpose=transpose, turnover=turnover))
        ...         ax.axis("off")
        ...         ax.set_title(f"transpose: {transpose}, turnover: {turnover}", fontsize=18)
        >>> fig.show()
    """
    if transpose:
        width, height = (height, width)
    indicator = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    for i in range(height):
        indicator[i] = score2color(score=i / height, cmap=cmap)
    if transpose:
        indicator = indicator.transpose(1, 0, 2)
    if turnover:
        indicator = indicator[::-1]
    return indicator


def drawScoreArc(
    frame: npt.NDArray[np.uint8],
    score: float,
    coords: Tuple[List[int], List[int], List[int]],
    inplace: bool = True,
    axes: Tuple[int, int] = (10, 10),
    lineType: int = cv2.LINE_8,
    cmap: Union[str, Colormap] = "coolwarm_r",
    max_score: Optional[float] = None,
    **kwargs,
) -> npt.NDArray[np.uint8]:
    """Draw an arc with fill color according to the ``score``.

    Args:
        frame (npt.NDArray[np.uint8])                 : Input image.
        score (float)                                 : Score value to describe.
        coords (Tuple[List[int],List[int],List[int]]) : Coordinates of the ``3`` points used to calculate the angle.
        inplace (bool, optional)                      : Whether frame is edited in place. Defaults to ``True``.
        axes (Tuple[int, int], optional)              : Half of the size of the ellipse main axes. Defaults to ``(10, 10)``.
        lineType (int, optional)                      : Type of the ellipse boundary. Defaults to ``cv2.LINE_8``.

    Returns:
        npt.NDArray[np.uint8]: An edited image.

    .. plot::
        :class: popup-img

        >>> import cv2
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from ddrev.utils import drawScoreArc, calculate_angle
        >>> fig, ax = plt.subplots()
        >>> A = np.asarray([0.2, 0.9])
        >>> B = np.asarray([0.8, 0.6])
        >>> C = np.asarray([0.3, 0.5])
        >>> frame = np.zeros(shape=(150, 100, 3), dtype=np.uint8)
        >>> H, W = frame.shape[:2]
        >>> drawScoreArc(frame, calculate_angle(A,B,C), coords=(A,B,C), max_score=360.)
        >>> drawScoreArc(frame, calculate_angle(A,C,B), coords=(A,C,B), max_score=360.)
        >>> drawScoreArc(frame, calculate_angle(B,A,C), coords=(B,A,C), max_score=360.)
        >>> pX, pY = (None, None)
        >>> for name, (x, y) in zip(list("ABCA"), [A,B,C,A]):
        ...     X, Y = (int(x * W), int(y * H))
        ...     ax.scatter(X, Y, color="red")
        ...     ax.text(x=X, y=Y - 10, s=name, size=20, color="red")
        ...     if pX is not None:
        ...         cv2.line(frame, (pX, pY), (X, Y), (255, 0, 0))
        ...     pX, pY = (X, Y)
        >>> ax.imshow(frame)
        >>> ax.axis("off")
        >>> ax.set_title("drawScoreArc", fontsize=18)
        >>> fig.show()
    """
    H, W = frame.shape[:2]
    if not inplace:
        frame = frame.copy()
    coords = np.asarray(coords)[:, :2]
    cx, cy = coords[1]
    cx_slide = (cx + 10, cy)  # Slide the center point in the x-axis direction
    startAngle = 360.0 - calculate_angle(cx_slide, *coords[1:])
    angle = calculate_angle(*coords)
    score = abs(score)
    if (score > 1) and (max_score is not None):
        score /= max_score
    cv2.ellipse(
        img=frame,
        center=(int(cx * W), int(cy * H)),
        axes=axes,
        angle=startAngle,
        startAngle=0,
        endAngle=angle,
        color=score2color(score, cmap=cmap),
        thickness=-1,
        lineType=lineType,
    )
    return frame


def drawAuxiliaryAngle(
    frame: npt.NDArray[np.uint8],
    score: float,
    coords: Tuple[List[int], List[int]],
    inplace: bool = True,
    thickness: int = 1,
    lineType: int = cv2.LINE_8,
    shift: int = 0,
    tipLength: float = 0.1,
    cmap: Union[str, Colormap] = "coolwarm_r",
    **kwargs,
) -> npt.NDArray[np.uint8]:
    """Draw an auxiliary arrow.

    Args:
        frame (npt.NDArray[np.uint8])         : Input image.
        score (float)                         : ``instructor's angles`` -``target's angles``.
        coords (Tuple[List[int], List[int]])  : Coordinates of the ``2`` points used to calculate the angle.
        inplace (bool, optional)              : Whether frame is edited in place. Defaults to ``True``.
        thickness (int, optional)             : [description]. Defaults to ``1``.
        lineType (int, optional)              : Type of the ellipse boundary. Defaults to ``cv2.LINE_8``.
        shift (int, optional)                 : [description]. Defaults to ``0``.
        tipLength (float, optional)           : [description]. Defaults to ``0.1``.
        cmap (Union[str, Colormap], optional) : [description]. Defaults to ``"coolwarm_r"``.

    Returns:
        npt.NDArray[np.uint8]: An edited image.

    .. plot::
        :class: popup-img

    Examples:
        >>> import cv2
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from ddrev.utils import drawAuxiliaryAngle
        >>> fig, ax = plt.subplots()
        >>> A = np.asarray([0.2, 0.9])
        >>> B = np.asarray([0.8, 0.6])
        >>> C = np.asarray([0.3, 0.5])
        >>> frame = np.zeros(shape=(150, 100, 3), dtype=np.uint8)
        >>> H, W = frame.shape[:2]
        >>> drawAuxiliaryAngle(frame,  0.3, coords=(B,C))
        >>> drawAuxiliaryAngle(frame, -0.9, coords=(A,C))
        >>> pX, pY = (None, None)
        >>> for name, (x, y) in zip(list("ABCA"), [A,B,C,A]):
        ...     X, Y = (int(x * W), int(y * H))
        ...     ax.scatter(X, Y, color="red")
        ...     ax.text(x=X, y=Y - 10, s=name, size=20, color="red")
        ...     if pX is not None:
        ...         cv2.line(frame, (pX, pY), (X, Y), (255, 0, 0))
        ...     pX, pY = (X, Y)
        >>> ax.imshow(frame)
        >>> ax.axis("off")
        >>> ax.set_title("drawAuxiliaryAngle", fontsize=18)
        >>> fig.show()
    """
    H, W = frame.shape[:2]
    if not inplace:
        frame = frame.copy()
    ax, ay = coords[0]
    cx, cy = np.mean(coords, axis=0)
    vx = ax - cx
    vy = ay - cy
    if score < 0:
        vy = -vy
    else:
        vx = -vx
    frame = cv2.arrowedLine(
        img=frame,
        pt1=(int(cx * W), int(cy * H)),
        pt2=(int((cx + vy) * W), int((cy + vx) * H)),
        color=score2color(abs(score), cmap=cmap),
        thickness=thickness,
        line_type=lineType,
        shift=shift,
        tipLength=tipLength,
    )
    return frame


def putScoreText(
    frame: npt.NDArray[np.uint8],
    score: float,
    coords: Tuple[List[int], List[int], List[int]],
    inplace: bool = True,
    fontFace: int = cv2.FONT_HERSHEY_PLAIN,
    fontScale: int = 1,
    color: Tuple[int, int, int] = (0, 255, 255),
    **kwargs,
) -> npt.NDArray[np.uint8]:
    """Write the ``score`` at the midpoint of both ends of the coordinates (``coords``).

    Args:
        frame (npt.NDArray[np.uint8])                 : Input image.
        score (float)                                 : Score value to describe.
        coords (Tuple[List[int],List[int],List[int]]) : Coordinates of the ``3`` points used to calculate the angle.
        inplace (bool, optional)                      : Whether frame is edited in place. Defaults to ``True``.
        fontFace (int, optional)                      : Font type. Defaults to ``cv2.FONT_HERSHEY_PLAIN``.
        fontScale (int, optional)                     : Font scale factor that is multiplied by the font-specific base size.. Defaults to ``2``.
        color (Tuple[int,int,int], optional)          : Text color. Defaults to ``(0,255,255)``.

    Returns:
        npt.NDArray[np.uint8]: An edited image.

    .. plot::
        :class: popup-img

        >>> import cv2
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from ddrev.utils import putScoreText, calculate_angle
        >>> fig, ax = plt.subplots()
        >>> coords = [
        ...     np.asarray([0.2, 0.9]),
        ...     np.asarray([0.8, 0.6]),
        ...     np.asarray([0.3, 0.5]),
        >>> ]
        >>> frame = np.zeros(shape=(150, 100, 3), dtype=np.uint8)
        >>> H, W = frame.shape[:2]
        >>> putScoreText(frame, calculate_angle(*coords), coords=coords)
        >>> pX, pY = (None, None)
        >>> for name, (x, y) in zip(list("ABC"), coords):
        ...     X, Y = (int(x * W), int(y * H))
        ...     ax.scatter(X, Y, color="red")
        ...     ax.text(x=X, y=Y - 10, s=name, size=20, color="red")
        ...     if pX is not None:
        ...         cv2.line(frame, (pX, pY), (X, Y), (255, 0, 0))
        ...     pX, pY = (X, Y)
        >>> ax.imshow(frame)
        >>> ax.axis("off")
        >>> ax.set_title("putScoreText", fontsize=18)
        >>> fig.show()
    """
    H, W = frame.shape[:2]
    if not inplace:
        frame = frame.copy()
    x, y = np.mean(coords, axis=0)[:2].tolist()
    cv2.putText(
        img=frame,
        text=f"{score:.1f}",
        org=(int(x * W), int(y * H)),
        fontFace=fontFace,
        fontScale=fontScale,
        color=color,
    )
    return frame
