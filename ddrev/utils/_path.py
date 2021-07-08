# coding: utf-8
import os

__all__ = [
    "UTILS_DIR",
    "MODULE_DIR",
    "REPO_DIR",
    "DATA_DIR",
    "SAMPLE_IMAGE",
    "SAMPLE_VIDEO",
]

UTILS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # path/to/Arcade-Dancing-System/ddrev/utils
MODULE_DIR = os.path.dirname(UTILS_DIR)  # path/to/Arcade-Dancing-System/ddrev
REPO_DIR = os.path.dirname(MODULE_DIR)  # path/to/Arcade-Dancing-System
DATA_DIR = os.path.join(REPO_DIR, "data")  # path/to/Arcade-Dancing-System/data
SAMPLE_IMAGE = os.path.join(
    DATA_DIR, "sample.jpeg"
)  # path/to/Arcade-Dancing-System/data/sample.jpeg"
SAMPLE_VIDEO = os.path.join(
    DATA_DIR, "sample-instructor.mp4"
)  # path/to/Arcade-Dancing-System/data/sample-instructor.mp4
