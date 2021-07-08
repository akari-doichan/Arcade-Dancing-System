import cv2
import matplotlib.pyplot as plt
from ddrev.poses.mediapipe import mpPoseEstimator
from ddrev.utils import drawScoreArc, putScoreText
from ddrev.utils._path import SAMPLE_IMAGE
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
image = cv2.imread(SAMPLE_IMAGE)  # BGR images
estimator = mpPoseEstimator()
landmarks = estimator.process(image)
estimator.draw_landmarks(image, landmarks, inplace=True)
scores = [(i+1)/len(landmarks.landmark) for i in range(len(landmarks.landmark))]
for draw_func, ax in zip([drawScoreArc, putScoreText], axes):
    img = estimator.draw_score(
        frame=image, scores=scores, landmarks=landmarks,
        draw_func=draw_func, inplace=False
    )
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(draw_func.__name__, fontsize=18)
fig.show()
