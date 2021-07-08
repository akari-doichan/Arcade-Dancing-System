import cv2
import numpy as np
import matplotlib.pyplot as plt
from ddrev.utils import drawScoreArc, calculate_angle
fig, ax = plt.subplots()
A = np.asarray([0.2, 0.9])
B = np.asarray([0.8, 0.6])
C = np.asarray([0.3, 0.5])
frame = np.zeros(shape=(150, 100, 3), dtype=np.uint8)
H, W = frame.shape[:2]
drawScoreArc(frame, calculate_angle(A,B,C), coords=(A,B,C), max_score=360.)
drawScoreArc(frame, calculate_angle(A,C,B), coords=(A,C,B), max_score=360.)
drawScoreArc(frame, calculate_angle(B,A,C), coords=(B,A,C), max_score=360.)
pX, pY = (None, None)
for name, (x, y) in zip(list("ABCA"), [A,B,C,A]):
    X, Y = (int(x * W), int(y * H))
    ax.scatter(X, Y, color="red")
    ax.text(x=X, y=Y - 10, s=name, size=20, color="red")
    if pX is not None:
        cv2.line(frame, (pX, pY), (X, Y), (255, 0, 0))
    pX, pY = (X, Y)
ax.imshow(frame)
ax.axis("off")
ax.set_title("drawScoreArc", fontsize=18)
fig.show()
