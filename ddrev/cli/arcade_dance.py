# coding: utf-8
import argparse
import json
import os
import sys
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .. import poses
from ..realtime import VideoCapture
from ..recorder import copyVideoSpec
from ..utils._colorings import toBLUE, toGREEN
from ..utils.feedback_utils import cmap_indicator_create, drawScoreArc
from ..utils.generic_utils import ListParamProcessor
from ..utils.score_utils import calculate_angle


def arcade_dance(argv=sys.argv[1:]):
    """Dance with instructor's video.

    Args:
        -J/--json (str)                    : A path to an instructor video json file.
        -C/--cam (int, optional)           : Your camera device number. Defaults to ``0``.
        -O/--out (str, optional)           : A path to an output video path. This argument is needed when ``record`` option is ``True``. Defaults to ``None``.
        --codec (str, optional)            : Codec of output video. This argument is needed when ``record`` option is ``True``. Defaults to ``"MP4V"``.
        --record (bool, optional)          : Whether to record or not. Defaults to ``False``.
        --axes (List[int])                 : Half of the size of the ellipse main axes. Defaults to ``[30, 30]``.
        --max-score (flaot, optional)      : If the score (angle difference with the instructor) is higher than this value, it will be rounded to this value. Defaults to ``90.``.
        --cmap-xywh (List[int], optional)  : The size and location of color map indicator. Defaults to ``[30, 30, 100, 200]``.
        --cmap-name (str)                  : The name of color map for score. Defaults to ``"coolwarm"``.
        --cmap-transpose (bool)            : Whether to transpose a color map indicator or not. Defaults to ``False``.

    NOTE:
        When you run from the command line, execute as follows::

        $ arcade-dance -V data/sample-instructor.json \\
                       --codec MP4V \\
                       --record H264 \\
                       --max-score 90
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
        default="coolwarm",
        help="The name of Color Map for Score. Defaults to coolwarm",
    )
    parser.add_argument(
        "--cmap-transpose",
        action="store_true",
        help="Whether to transpose a color map indicator or not.",
    )
    args = parser.parse_args(argv)

    with open(args.json) as f:
        data = json.load(f)
    cap = VideoCapture(args.cam)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # record = args.record
    # is_ok, out, out_path = copyVideoSpec(cap=cap, codec=args.codec, out_path=args.out)
    # if record and (not is_ok):
    #     raise TypeError(f"VideoWriter was not created correctly.")

    model = data["model"]
    video_path = data["video"]
    score_method = data["score_method"]
    angle_points = data["angle_points"]
    angle_unit = data["angle_unit"]
    instructor_frame_count = data["frame_count"]
    instructor_landmarks = data["landmarks"]
    instructor_scores = np.asarray(data["scores"], dtype=float)

    video = cv2.VideoCapture(video_path)
    estimator = poses.get(identifier=model)
    axes = args.axes

    # Color Map Indicator
    cmap_x, cmap_y, cmap_w, cmap_h = args.cmap_xywh
    cmap_transpose = args.cmap_transpose
    cmap_indicator = cmap_indicator_create(
        width=cmap_w,
        height=cmap_h,
        transpose=cmap_transpose,
        turnover=True,
        cmap=args.cmap_name,
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
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        # Students
        # bg = np.zeros(shape=(width, height, 3), dtype=np.uint8)
        landmarks = estimator.process(frame)
        scores = estimator.calculate_angle(
            landmarks=landmarks, angle_points=angle_points, unit=angle_unit
        )
        frame = estimator.draw_landmarks(frame=frame, landmarks=landmarks)

        # Instructors
        curt_idx = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        is_ok, frame_ = video.read()
        if (not is_ok) or (frame_ is None) or (curt_idx >= instructor_frame_count):
            return frame
        landmarks_ = estimator.string2landmarks(instructor_landmarks[curt_idx])
        scores_ = instructor_scores[curt_idx]
        frame_ = estimator.draw_landmarks(frame=frame_, landmarks=landmarks_)
        cv2.imshow(winname="Instructors", mat=frame_)

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
