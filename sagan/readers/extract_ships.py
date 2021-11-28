from sagan import config

from ipdb import set_trace as debug
from glob import glob
import imutils
import numpy as np
import pylab as plt
from skimage import io
from skimage.color import rgb2gray
from tqdm import tqdm
from sagan.utils.vessel import Vessel
import xmltodict


def extract_training_data():
    """Extract annotation data from collections of XML files."""
    path_to_annotations = f"{config.TL_DATA_PATH}/training/annotations/*.xml"
    xml_files = glob(path_to_annotations)
    return xml_files


def extract_tiles(M, ship, tile_size=150, nb_tiles=10, augment_via_rotation=True):
    """Extract @nb_tiles random tiles from the matrix @M."""
    yt = int(ship["bndbox"]["ymin"])
    yb = int(ship["bndbox"]["ymax"])
    xl = int(ship["bndbox"]["xmin"])
    xr = int(ship["bndbox"]["xmax"])
    col_ = int((xr + xl) / 2)
    row_ = int((yt + yb) / 2)
    dy = yb - yt
    dx = xr - xl
    half_width = tile_size // 2
    radius = int(half_width * np.sqrt(2))
    height, width = M.shape
    # Not enough space to extract requested tile — return nothing.
    if (
        2 * radius >= width
        or 2 * radius >= height
        or not is_extractable(row_, col_, radius, M)
    ):
        return []
    rows = np.random.randint(radius, height - radius, nb_tiles)
    cols = np.random.randint(radius, width - radius, nb_tiles)
    rows = nb_tiles * [row_]
    cols = nb_tiles * [col_]
    tiles = []
    for idx in range(nb_tiles):
        row, col = rows[idx], cols[idx]
        rotatable_tile = M[row - radius : row + radius, col - radius : col + radius]
        if augment_via_rotation and np.random.rand() > 0.0:
            rotatable_tile = imutils.rotate(rotatable_tile, np.random.uniform(15, 335))
        low = radius - tile_size // 2
        high = radius + tile_size // 2
        tile = rotatable_tile[low:high, low:high]
        tiles.append(tile)
    return tiles


def is_extractable(row, col, radius, M):
    """Can we extract the specified tile from the @image?"""
    height, width = M.shape
    # We are optimistic: we probably can!
    extractable = True
    # And yet we are also practical: trust, but verify.
    extractable *= row - radius > 0
    extractable *= row + radius < height
    extractable *= col - radius > 0
    extractable *= col + radius < width
    return extractable


def extract_ship(ship, image, chip_size=128):
    yt = int(ship["bndbox"]["ymin"])
    yb = int(ship["bndbox"]["ymax"])
    xl = int(ship["bndbox"]["xmin"])
    xr = int(ship["bndbox"]["xmax"])
    dy = yb - yt
    dx = xr - xl
    x_pad = np.round((128 - dx) / 2).astype("int")
    y_pad = np.round((128 - dy) / 2).astype("int")

    small_image = image[yt:yb, xl:xr]
    return small_image, (yb - yt, xr - xl)


if __name__ == "__main__":
    xf = extract_training_data()
    path_to_images = f"{config.TL_DATA_PATH}/training/JPEGImages"

    plt.close("all")
    plt.ion()

    ships = []
    tiles = []
    # idx = np.random.randint(len(xf))
    # for xml in xf[idx : idx + 1]:
    for xml in tqdm(xf):
        with open(xml, "r") as f:
            data = xmltodict.parse(f.read())
            ship = {}
            filename = data["annotation"]["filename"]
            image = rgb2gray(io.imread(f"{path_to_images}/{filename}"))
            if type(data["annotation"]["object"]) == list:
                # There are more than one ship in this image>
                for obj in data["annotation"]["object"]:
                    images = extract_tiles(image, obj, nb_tiles=3)
                    tiles += images
            else:
                ship = data["annotation"]["object"]
                tiles += extract_tiles(image, ship)

    if len(tiles) > 0:
        idx = np.random.randint(len(tiles))
        plt.imshow(tiles[idx], cmap="gray")
        v = Vessel("ship_images.dat")
        v.tiles = tiles
        v.save()
