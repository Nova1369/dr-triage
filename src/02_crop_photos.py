"""Step 2. Crop every photo to its retina and shrink it to 224 x 224 -> data/crops.npy

A fundus photo is a bright disc (the retina) on a black background, and every camera leaves a different
amount of black around it. For each photo we

  1. find the rows and columns that contain retina (pixels brighter than the black background),
  2. cut the photo down to that box,
  3. pad the box with black to a square, so the retina keeps its shape,
  4. resize it to 224 x 224 pixels, the input size of the network used in step 3.

Colours are not changed. For each photo we also note its original width and height (they tell cameras
apart), its brightness, its sharpness and how much of the crop the retina fills (a check on the cropping).

All crops go into one array, data/crops.npy, so that later steps never have to open the large original
files again. The script can be stopped at any time: run it again and it continues where it stopped.

Run from the project folder:   python src/02_crop_photos.py
"""
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SIZE = 224          # side of the square crop, in pixels
BACKGROUND = 20     # a pixel is "retina" if its brightest colour channel is above this (0-255)
SAVE_EVERY = 100    # photos between two saves of the progress

DATASETS = Path("datasets")
OUT = Path("data")
CROPS_FILE = OUT / "crops.npy"
STATS_FILE = OUT / "photo_stats.csv"
STATS_COLUMNS = ["width", "height", "brightness", "sharpness", "retina_fill"]


def crop_retina(image):
    """Photo (PIL image) -> SIZE x SIZE x 3 array in which the retina fills the frame."""
    image.thumbnail((1024, 1024))                      # a smaller copy is faster and plenty for 224 pixels
    pixels = np.asarray(image.convert("RGB"))
    retina = pixels.max(axis=2) > BACKGROUND            # True on the retina, False on the black background
    rows = np.where(retina.mean(axis=1) > 0.05)[0]      # rows and columns in which at least 5 % is retina
    cols = np.where(retina.mean(axis=0) > 0.05)[0]
    if len(rows) > 0 and len(cols) > 0:
        pixels = pixels[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    height, width = pixels.shape[:2]
    side = max(height, width)
    square = np.zeros((side, side, 3), dtype=np.uint8)  # black square, with the box pasted in its centre
    top, left = (side - height) // 2, (side - width) // 2
    square[top:top + height, left:left + width] = pixels
    return np.asarray(Image.fromarray(square).resize((SIZE, SIZE), Image.Resampling.LANCZOS))


def sharpness(crop):
    """Variance of the difference between each pixel and its four neighbours: high for crisp photos."""
    grey = crop.mean(axis=2)
    difference = 4 * grey[1:-1, 1:-1] - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
    return float(difference.var())


def process(path):
    """One photo: its crop and its row of notes (STATS_COLUMNS)."""
    with Image.open(DATASETS / path) as image:
        width, height = image.size
        crop = crop_retina(image)
    retina_fill = float((crop.max(axis=2) > BACKGROUND).mean())
    return crop, [width, height, float(crop.mean()), sharpness(crop), retina_fill]


if __name__ == "__main__":
    photos = pd.read_csv(OUT / "photos.csv")
    n = len(photos)

    if CROPS_FILE.exists() and STATS_FILE.exists():     # continue an earlier run
        crops = np.load(CROPS_FILE, mmap_mode="r+")
        stats = pd.read_csv(STATS_FILE).values.tolist()
    else:
        crops = np.lib.format.open_memmap(CROPS_FILE, mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
        stats = []

    def save():
        crops.flush()
        pd.DataFrame(stats, columns=STATS_COLUMNS).to_csv(str(STATS_FILE) + ".tmp", index=False)
        os.replace(str(STATS_FILE) + ".tmp", STATS_FILE)

    done = len(stats)
    print(f"{done} of {n} photos already cropped")
    with Pool() as pool:                                 # one worker per processor core
        for crop, row in pool.imap(process, photos["path"][done:], chunksize=4):
            crops[len(stats)] = crop
            stats.append(row)
            if len(stats) % SAVE_EVERY == 0:
                save()
                print(f"{len(stats)} of {n}", flush=True)
    save()

    table = pd.concat([photos[["dataset"]], pd.DataFrame(stats, columns=STATS_COLUMNS)], axis=1)
    print("\nMedian per dataset:")
    print(table.groupby("dataset")[["brightness", "sharpness", "retina_fill"]].median().round(2))
    print(f"\nPhoto sizes (cameras) in APTOS: {table[table.dataset == 'aptos'].groupby(['width', 'height']).ngroups}")
    print(f"Crops with the retina filling less than half of the frame: {(table.retina_fill < 0.5).sum()}")
    print(f"\n{n} crops in {CROPS_FILE}, notes in {STATS_FILE}")
