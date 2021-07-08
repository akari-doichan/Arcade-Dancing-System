import cv2
import matplotlib.pyplot as plt
from ddrev.poses.mediapipe import mpPoseEstimator
from ddrev.utils._path import SAMPLE_IMAGE
image = cv2.imread(SAMPLE_IMAGE) # BGR images
estimator = mpPoseEstimator()
image_edited = estimator.process_frame(image, inplace=False)
fig, axes = plt.subplots(ncols=2, figsize=(12,4))
for ax,img,title in zip(axes, [image, image_edited], ["Original", "Pose-Estimated"]):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(title, fontsize=18)
fig.show()
