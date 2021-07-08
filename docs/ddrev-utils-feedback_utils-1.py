import numpy as np
import matplotlib.pyplot as plt
from ddrev.utils import cmap_indicator_create
fig, axes = plt.subplots(ncols=2,nrows=2,figsize=(12,8))
for ax_r,transpose in zip(axes, [False, True]):
    for ax,turnover in zip(ax_r, [False, True]):
        ax.imshow(cmap_indicator_create(width=100, height=50, transpose=transpose, turnover=turnover))
        ax.axis("off")
        ax.set_title(f"transpose: {transpose}, turnover: {turnover}", fontsize=18)
fig.show()
