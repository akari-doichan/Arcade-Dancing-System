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
    args = parser.parse_args(argv)

    video_path = os.path.abspath(args.video)
    out_path = args.out
    model = args.model
    if out_path is None:
        out_path = f"{os.path.splitext(video_path)[0]}_{model}.json"
    # Create an instance of Pose Estimator.
    estimator = poses.get(identifier=model)
    # Capture an input video.
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"""[Video -> Landmarks]
    * Video: {toBLUE(video_path)}
        * Width : {toGREEN(width)}
        * Height: {toGREEN(height)}
        * Count : {toGREEN(frame_count)}
    * Model: {toGREEN(model)}
    Landmarks file will be saved at {toBLUE(out_path)}
    """
    )

    # Collect Landmarks information for each frame in video.
    landmarks = []
    for i in tqdm(range(frame_count), desc="Video -> JSON"):
        is_ok, frame = cap.read()
        if (not is_ok) and (frame is None):
            break
        ith_landmarks = estimator.process(frame=frame)
        landmarks.append(estimator.landmarks2string(ith_landmarks))
    cap.release()

    if hasattr(estimator, "__exit__"):
        estimator.close()

    # Save Landmarks information with video metadata.
    save_json(
        obj={
            "model": model,
            "video": video_path,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "landmarks": landmarks,
        },
        file=out_path,
        indent=2,
    )
    print(f"Landmarks file was saved at {toBLUE(out_path)} correctly.")
