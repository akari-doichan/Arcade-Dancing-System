#coding: utf-8
import os

__all__ = [
    "UTILS_DIR", "MODULE_DIR", "REPO_DIR",
]

UTILS_DIR       = os.path.dirname(os.path.abspath(__file__)) # path/to/Arcade-Dancing-System/ddrev/utils
MODULE_DIR      = os.path.dirname(UTILS_DIR)                 # path/to/Arcade-Dancing-System/ddrev
REPO_DIR        = os.path.dirname(MODULE_DIR)                # path/to/Arcade-Dancing-System
