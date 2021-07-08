# coding: utf-8
import argparse
import json
import os
import sys
import warnings
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .. import poses
from ..realtime import VideoCapture
from ..recorder import copyVideoSpec
from ..utils._colorings import toBLUE, toGREEN, toRED
from ..utils.feedback_utils import cmap_indicator_create, drawScoreArc
from ..utils.generic_utils import ListParamProcessor, now_str
from ..utils.score_utils import calculate_angle


def arcade_dance(argv=sys.argv[1:]):
    """Dance with instructor's video.

    Args:
        -J/--json (str)                         : A path to an instructor video json file.
        -C/--cam (int, optional)                : Your camera device number. Defaults to ``0``.
        --instructor-xywh (List[int], optional)  : The size and location of instructor video. Defaults to ``None``.
        --axes (List[int])                      : Half of the size of the ellipse main axes. Defaults to ``[30, 30]``.
        --max-score (flaot, optional)           : If the score (angle difference with the instructor) is higher than this value, it will be rounded to this value. Defaults to ``90.``.
        --cmap-xywh (List[int], optional)       : The size and location of color map indicator. Defaults to ``[30, 30, 100, 200]``.
        --cmap-name (str)                       : The name of color map for score. Defaults to ``"coolwarm_r"``.
        --cmap-transpose (bool)                 : Whether to transpose a color map indicator or not. Defaults to ``False``.
        -O/--out (str, optional)                : A path to an output video path. This argument is needed when ``record`` option is ``True``. Defaults to ``None``.
        --codec (str, optional)                 : Codec of output video. This argument is needed when ``record`` option is ``True``. Defaults to ``"MP4V"``.
        --record (bool, optional)               : Whether to record or not. Defaults to ``False``.

    NOTE:
        When you run from the command line, execute as follows::

        $ arcade-dance -J data/sample-instructor_mediapipe_angle.json \\
                       --max-score 90 \\
                       --instructor-xywh "[-410,10,400,400]" \\
                       --codec MP4V \\
                       --record
    """
    parser = argparse.ArgumentParser(
        prog="dance",
        description="Dance with instructor's video.",
        add_help=True,
    )
    parser.add_argument(
        "-J",
        "--json",
        type=str,
        required=True,
        help="A path to an instructor video json file.",
    )
    parser.add_argument(
        "-C",
        "--cam",
        type=int,
        default=0,
        help="Your camera device number. Defaults to 0.",
    )
    parser.add_argument(
        "--instructor-xywh",
        action=ListParamProcessor,
        default=None,
        help="The size and location of instructor video. Defaults to None.",
    )
    parser.add_argument(
        "--axes",
        action=ListParamProcessor,
        default=[30, 30],
        help="Half of the size of the ellipse main axes. Defaults to [30, 30]",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=90.0,
        help="If the score is higher than this value (angle difference with the instructor), it will be rounded to this value.",
    )
    parser.add_argument(
        "--cmap-xywh",
        action=ListParamProcessor,
        default=[30, 30, 100, 200],
        help="Size and location of Color Map Indicator. Defaults to [30, 30, 100, 200]",
    )
    parser.add_argument(
        "--cmap-name",
        type=str,
        default="coolwarm_r",
        help="The name of Color Map for Score. Defaults to coolwarm",
    )
    parser.add_argument(
        "--cmap-transpose",
        action="store_true",
        help="Whether to transpose a color map indicator or not.",
    )
    parser.add_argument(
        "-O", "--out", type=str, default=None, help="A path to an output video file."
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="MP4V",
        help="Codec of output video. Defaults to MP4V.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Whether to record or not.",
    )
    args = parser.parse_args(argv)

    with open(args.json) as f:
        data = json.load(f)

    # Instructor7s video
    video_path = data["video"]
    video = cv2.VideoCapture(video_path)
    instructor_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    instructor_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    instructor_fps = video.get(cv2.CAP_PROP_FPS)
    instructor_frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    instructor_duration_sec = instructor_frame_count / instructor_fps
    instructor_landmarks = data["landmarks"]
    instructor_scores = np.asarray(data["scores"], dtype=float)
    instructor_xywh = args.instructor_xywh
    if instructor_xywh is not None:
        integrate_windows = True
        inst_x, inst_y, inst_w, inst_h = args.instructor_xywh
    else:
        integrate_windows = False
        inst_x, inst_y, inst_w, inst_h = (0, 0, instructor_width, instructor_height)

    record = args.record
    if record and (instructor_xywh is None):
        warnings.warn(
            f"If you specify {toGREEN('--instructor-xywm')} argument, the instructor's video can also be drawn in the same frame."
        )

    cam = args.cam
    out_path = args.out
    if record and out_path is None:
        out_path = f"arcade-dance_{now_str()}.mp4"
    out_codec = args.codec
    cap = VideoCapture(cam, out_path=out_path, codec=out_codec)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Estimator
    model = data["model"]
    score_method = data["score_method"]
    angle_points = data["angle_points"]
    angle_unit = data["angle_unit"]
    estimator = poses.get(identifier=model)
    axes = args.axes

    # Color Map Indicator
    cmap_name = args.cmap_name
    cmap_x, cmap_y, cmap_w, cmap_h = args.cmap_xywh
    cmap_transpose = args.cmap_transpose
    cmap_indicator = cmap_indicator_create(
        width=cmap_w,
        height=cmap_h,
        transpose=cmap_transpose,
        turnover=True,
        cmap=cmap_name,
    )
    max_score = args.max_score
    cmap_org_max_min = (
        [
            (int(cmap_x - 10), int(cmap_y + cmap_h // 2)),
            (int(cmap_x + cmap_w + 10), int(cmap_y + cmap_h // 2)),
        ]
        if cmap_transpose
        else [
            (int(cmap_x + cmap_w // 2), int(cmap_y - 10)),
            (int(cmap_x + cmap_w // 2), int(cmap_y + cmap_h + 10)),
        ]
    )
    cmap_color_max_min = [
        tuple([int(e) for e in cmap_indicator[0, 0]]),
        tuple([int(e) for e in cmap_indicator[-1, -1]]),
    ]

    msg = f"""[Arcade Dancing System]
    * Model: {toGREEN(model)}
    * Scoring Method: {toGREEN(score_method)}
    * Web Camera
        * Camera ID : {toGREEN(cam)}
        * Size (W,H): ({toGREEN(width)},{toGREEN(height)})
    * Instructor Video: {toBLUE(video_path)}
        * Frame Count: {toGREEN(instructor_frame_count)}
        * FPS        : {toGREEN(f'{instructor_fps:.1f}')}
        * Duration   : {toGREEN(f'{instructor_duration_sec:.1f}[s]')}
        * Size (W,H) : ({toGREEN(instructor_width)},{toGREEN(instructor_height)})
        * Resize to -> ({toGREEN(inst_w)},{toGREEN(inst_h)}) @ (x={toGREEN(inst_x)},y={toGREEN(inst_y)})
    * Color Map: {toGREEN(cmap_name)}
        * Size (W,H) : ({toGREEN(cmap_w)},{toGREEN(cmap_h)}) @ (x={toGREEN(cmap_x)},y={toGREEN(cmap_y)})
    * Key Operation
        * {toGREEN('r')} -> Restart the instructor video.
        * {toGREEN('d')} -> Increase the speed of the instructor video.
        * {toGREEN('s')} -> Slow donw the speed of the instructor video.
        * {toGREEN('q')} or {toGREEN('<esc>')} -> Stop the program."""
    if record:
        msg += f"""
    * Record ({toRED("DO NOT exit this program with control+C")})
        * Output Path: {toBLUE(cap.out_path)}
    """
    print(msg)

    VIDEO_SPEED = [1]

    def process(frame: npt.NDArray[np.uint8], key: int) -> npt.NDArray[np.uint8]:
        """Process frame in while-loop in :meth:`realtime_process <ddrev.realtime.VideoCapture.realtime_process>`.

        Args:
            frame (npt.NDArray[np.uint8]) : A three channel ``BGR`` image represented as numpy ndarray.
            key (int, optional)           : An integer representing the Unicode character. Defaults to ``1``.

        Returns:
            npt.NDArray[np.uint8]: An edited (drawn ``landmarks`` and score) frame.
        """
        if key == ord("r"):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == ord("d"):
            VIDEO_SPEED[0] = VIDEO_SPEED[0] + 1
        elif key == ord("s"):
            VIDEO_SPEED[0] = max(VIDEO_SPEED[0] - 1, 1)

        # Students
        # bg = np.zeros(shape=(width, height, 3), dtype=np.uint8)
        landmarks = estimator.process(frame)
        scores = estimator.calculate_angle(
            landmarks=landmarks, angle_points=angle_points, unit=angle_unit
        )
        frame = estimator.draw_landmarks(frame=frame, landmarks=landmarks)

        # Instructors
        for _ in range(VIDEO_SPEED[0]):
            curt_idx = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            if curt_idx >= instructor_frame_count:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            is_ok, frame_ = video.read()
            if (not is_ok) or (frame_ is None):
                return frame
        landmarks_ = estimator.string2landmarks(instructor_landmarks[curt_idx])
        scores_ = instructor_scores[curt_idx]

        # Draw Score.
        estimator.draw_score(
            frame=frame,
            scores=np.abs(scores - scores_),
            landmarks=landmarks,
            draw_func=drawScoreArc,
            inplace=True,
            max_score=max_score,
            axes=axes,
        )

        # Draw Instructor's video
        frame_ = estimator.draw_landmarks(frame=frame_, landmarks=landmarks_)
        cv2.putText(
            img=frame_,
            text=f"{video.get(cv2.CAP_PROP_POS_MSEC)/1000:.2f}/{instructor_duration_sec:.1f}[s]",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            color=(0, 0, 255),
        )
        cv2.putText(
            img=frame_,
            text=f"Speed: {VIDEO_SPEED[0]}",
            org=(instructor_width - 150, 30),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            color=(0, 0, 255),
        )
        if integrate_windows:
            frame[inst_y : inst_y + inst_h, inst_x : inst_x + inst_w, :] = cv2.resize(
                frame_, dsize=(inst_w, inst_h)
            )
        else:
            cv2.imshow(winname="Instructors", mat=frame_)

        # Draw Color Map Indicator
        frame[cmap_y : cmap_y + cmap_h, cmap_x : cmap_x + cmap_w, :] = cmap_indicator
        for cmap_org, val, color in zip(
            cmap_org_max_min, [max_score, 0], cmap_color_max_min
        ):
            cv2.putText(
                img=frame,
                text=str(val),
                org=cmap_org,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=color,
            )
        return frame

    cap.realtime_process(function=process)
