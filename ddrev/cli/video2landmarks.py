# coding: utf-8
import argparse
import json
import os
import sys

import cv2
from tqdm import tqdm

from .. import poses
from ..utils._colorings import toBLUE, toGREEN
from ..utils.json_utils import save_json


def video2landmarks(argv=sys.argv[1:]):
    """Convert from video to landmarks data (``.json``)

    Args:
        -V/--video (str)                 : A path to an input Video file.
        -O/--out (str, optional)         : A path to an output json file. Defaults to ``None``.
        --model (str, optional)          : The Name of Pose-Estimation Model. Defaults to ``"mediapipe"``.
        --score-method (str, optional)   : How to calculate scores. Defaults to ``"angle"``.
        --angle-unit (str, optional)     : Unit of Angle. Defaults to ``"radian"``.
        --store-abspath (bool, optional) : Whether to keep the absolute path or relative path of the video file. Defaults to relative path.

    NOTE:
        When you run from the command line, execute as follows::

            $ video2landmarks -V path/to/video.mp4 \\
                              --model mediapipe \\
                              --score-method angle
    """
    parser = argparse.ArgumentParser(
        prog="video2landmarks",
        description="Convert from video to landmarks data.",
        add_help=True,
    )
    parser.add_argument(
        "-V", "--video", type=str, required=True, help="A path to an input Video file."
    )
    parser.add_argument(
        "-O", "--out", type=str, default=None, help="A path to an output json file."
    )
    parser.add_argument(
        "--model",
        choices=["mediapipe"],
        default="mediapipe",
        help="The Name of Pose-Estimation Model.",
    )
    parser.add_argument(
        "--score-method",
        choices=["angle", "distance"],
        default="angle",
        help="How to calculate scores.",
    )
    parser.add_argument(
        "--angle-unit",
        choices=["radian", "degree"],
        default="degree",
        help="Unit of Angle.",
    )
    parser.add_argument(
        "--store-abspath",
        actions="store_true",
        help="Whether to keep the absolute path or relative path of the video file. Defaults to relative path.",
    )
    args = parser.parse_args(argv)

    video_path = args.video
    if args.store_abspath:
        video_path = os.path.abspath(video_path)
    out_path = args.out
    model = args.model
    score_method = args.score_method
    angle_unit = args.angle_unit
    if out_path is None:
        out_path = f"{os.path.splitext(video_path)[0]}_{model}_{score_method}.json"
    # Capture an input video.
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    data = {
        "model": model,
        "video": video_path,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "score_method": score_method,
    }
    for k, v in data.items():
        print(f"* {k}: {toGREEN(k)}")
    print(f"Landmarks file will be saved at {toBLUE(out_path)}")

    # Create an instance of Pose Estimator.
    estimator = poses.get(identifier=model)
    if score_method == "angle":
        data["angle_points"] = estimator.ANGLE_POINTS
        data["angle_unit"] = angle_unit

    # Collect Landmarks information for each frame in video.
    landmarks = []
    scores = []
    for i in tqdm(range(frame_count), desc="Video -> JSON"):
        is_ok, frame = cap.read()
        if (not is_ok) and (frame is None):
            break
        ith_landmarks = estimator.process(frame=frame)
        landmarks.append(estimator.landmarks2string(ith_landmarks))
        if score_method == "angle":
            angle = estimator.calculate_angle(ith_landmarks, unit=angle_unit)
            scores.append(angle)
    cap.release()

    if hasattr(estimator, "__exit__"):
        estimator.close()

    # Save Landmarks information with video data.
    data.update({"landmarks": landmarks, "scores": scores})
    save_json(obj=data, file=out_path, indent=2)
    print(f"Landmarks file was saved at {toBLUE(out_path)} correctly.")
