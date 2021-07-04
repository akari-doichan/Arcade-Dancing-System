#coding: utf-8
import cv2
import warnings
from pydub import AudioSegment
from pydub.playback import play
from numbers import Number

from .utils._colorings import toRED

class PlayBack():
    def __init__(self, path:str):
        self.audio = AudioSegment.from_file(file=path, format="mp4")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened:
            warnings.warn(toRED(f"The video could not be loaded properly. Please check the {toGREEN('path')}."))
        else:
            self.synchronize()
        self.tm = cv2.TickMeter()
        self.winname = "hoge"

    def play(self):
        play(self.audio)
        while self.cap.isOpened():
            is_ok, frame = self.cap.read()
            if (not is_ok) or (frame is None):
                break
            while True:
                if self.tm.getTimeSec()>self.fps:
                    break
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    @property
    def fps():
        return self.cap.get(cv2.CAP_PROP_FPS)

    def set_fps(self, fps:Number, synchronize:bool = True):
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        if synchronize:
            self.synchronize()

    def synchronize(self) -> None:
        """Synchronize the audio with current video fps."""
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.audio = self.audio._spawn(self.audio.raw_data, overrides={
            "frame_rate": int(video_fps)
        })
        self.audio.set_frame_rate(self.audio.frame_rate)
