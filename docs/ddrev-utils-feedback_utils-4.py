import cv2
import numpy as np
import matplotlib.pyplot as plt
from ddrev.utils import putScoreText, calculate_angle
fig, ax = plt.subplots()
coords = [
    np.asarray([0.2, 0.9]),
    np.asarray([0.8, 0.6]),
    np.asarray([0.3, 0.5]),
]
frame = np.zeros(shape=(150, 100, 3), dtype=np.uint8)
H, W = frame.shape[:2]
putScoreText(frame, calculate_angle(*coords), coords=coords)
pX, pY = (None, None)
for name, (x, y) in zip(list("ABC"), coords):
    X, Y = (int(x * W), int(y * H))
    ax.scatter(X, Y, color="red")
    ax.text(x=X, y=Y - 10, s=name, size=20, color="red")
    if pX is not None:
        cv2.line(frame, (pX, pY), (X, Y), (255, 0, 0))
    pX, pY = (X, Y)
ax.imshow(frame)
ax.axis("off")
ax.set_title("putScoreText", fontsize=18)
fig.show()
