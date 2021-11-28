from sagan.utils.vessel import Vessel

import numpy as np
import pylab as plt


d = Vessel("ship_images.dat")
plt.ioff()

for idx, img in enumerate(d.tiles):
    if np.mod(idx + 1, 1) == 0:
        print(f"{idx/len(d.tiles)*100:2f}%")
    plt.close("all")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(f"ships/ship_{idx:05d}.jpg", bbox_inches="tight", pad_inches=0)
