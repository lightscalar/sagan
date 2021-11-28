from glob import glob
import numpy as np
import pylab as plt
from skimage import io

path_to_planes = "/Users/mjl/projects/army/sagan/sagan/data/PLANES/planesnet/planesnet"
path_to_ships = "/Users/mjl/projects/army/sagan/sagan/data/HRSID_JPG/JPEGImages"
files = glob(f"{path_to_ships}/*.jpg")


if __name__ == "__main__":

    # Load a sample image.
    idx = np.random.randint(len(files))
    image = io.imread(files[idx])

    plt.ion()
    plt.close("all")
    plt.imshow(image)
