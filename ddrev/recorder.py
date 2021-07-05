# coding: utf-8
import os
import warnings
from typing import Optional, Tuple

import cv2

from .utils._colorings import toGREEN, toRED
from .utils.generic_utils import now_str
from .utils.video_utils import videocodec2ext


def copyVideoSpec(
    cap: cv2.VideoCapture,
    codec: str = "MP4V",
    out_path: Optional[str] = None,
    fps: Optional[float] = None,
) -> Tuple[bool, cv2.VideoWriter, str]:
    """
    Args:
        cap (cv2.VideoCapture)             : An instance of ``cv2.VideoCapture``.
        codec (str, optional)              : Video Codec. Defaults to ``"MP4V"``.
        out_path (Optional[str], optional) : The path to which the ``cv2.VideoWriter`` writes video. Defaults to ``None``.
        fps (Optional[float], optional)    : fps for the output video. Defaults to ``None``.

    Returns:
        Tuple[bool, cv2.VideoWrier, str] : Tuple of flag (whether ``cv2.VideoWriter`` is correctly created), ``cv2.VideoWriter``, and path to output video.

    Examples:
        >>> import time
        >>> import cv2
        >>> from tqdm import tqdm
        >>> from ddrev.realtime import VideoCapture
        >>> from ddrev.recorder import copyVideoSpec
        >>> cap = cv2.VideoCapture(1)
        >>> is_ok, out, out_path = copyVideoSpec(cap=cap, codec="MP4V")
        >>> n = 100
        >>> digit = len(str(n))
        >>> if is_ok:
        ...     s = time.time()
        ...     for i in tqdm(range(n), desc="Recording"):
        ...         is_ok, frame = cap.read()
        ...         cv2.putText(
        ...             img=frame,
        ...             text=f"{i+1:>0{digit}}/{n}",
        ...             org=(10, 50),
        ...             fontFace=cv2.FONT_HERSHEY_PLAIN,
        ...             fontScale=3,
        ...             color=(255, 255, 255),
        ...         )
        ...         out.write(frame)
        ...     out.set(cv2.CAP_PROP_FPS, (time.time() - s) / n)
        ...     out.release()
        ...     print(f"Captured video was saved at {out_path}")
    """
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    ideal_ext = videocodec2ext(codec)
    if out_path is None:
        out_path = now_str() + ideal_ext
    else:
        root, original_ext = os.path.splitext(out_path)
        if original_ext != ideal_ext:
            root, original_ext = os.path.splitext(out_path)
            warnings.warn(
                f"Change the file extension from {toRED(original_ext)} to {toGREEN(ideal_ext)} according to video codec ({toGREEN(codec)})."
            )
            out_path = root + ideal_ext
    VideoWriter = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    is_ok = VideoWriter.isOpened()
    if not is_ok:
        warnings.warn(
            toRED(
                """\
        Could not make a typing video because VideoWriter was not created successfully.\n\
        Look at the warning text from OpenCV above and do what you need to do.\n\
        """
            )
        )
    return (is_ok, VideoWriter, out_path)
