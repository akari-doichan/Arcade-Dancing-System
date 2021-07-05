# coding: utf-8
from typing import Union


def videocodec2ext(*codec) -> str:
    """Convert video codec to video extension.

    Args:
        codec (Union[tuple, str]) : Video Codec.

    Returns:
        str: Ideal file extension.

    Examples:
        >>> from pycharmers.opencv import videocodec2ext
        >>> videocodec2ext("MP4V")
        '.mp4'
        >>> videocodec2ext("mp4v")
        '.mov'
        >>> videocodec2ext("VP80")
        '.webm'
        >>> videocodec2ext("XVID")
        '.avi
        >>> videocodec2ext("☺️")
        '.mp4'
    """
    if len(codec) == 1 and isinstance(codec[0], str):
        codec = codec[0]
    else:
        codec = "".join(codec)
    return {
        "VP80": ".webm",
        "MP4S": ".mp4",
        "MP4V": ".mp4",
        "mp4v": ".mov",
        "H264": ".mp4",
        "X264": ".mp4",
        "DIV3": ".avi",
        "DIVX": ".avi",
        "IYUV": ".avi",
        "MJPG": ".avi",
        "XVID": ".avi",
        "THEO": ".ogg",
        "H263": ".wmv",
    }.get(codec, ".mp4")
