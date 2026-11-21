"""Step 12. Features of worse photos, for the robustness test of step 13 -> data/features_<change>.npy

A screening camp does not always produce good photos. The APTOS crops of step 2 are made worse in three
ways, one at a time, and passed through the same frozen ResNet-50 as in step 3:

  blur   Gaussian blur with a radius of 1.3 pixels: a photo slightly out of focus
  dark   every pixel at half its brightness: an under-exposed photo
  jpeg   saved again as JPEG at quality 20: strong compression, as on a slow telemedicine link

Only the APTOS photos are changed (3,662 photos, in the order of data/table.csv), because step 13 tests
the changes where the rule works, at APTOS. Each file has 3,662 rows of 2,048 features.

The script can be stopped at any time: run it again and it continues where it stopped.

Run from the project folder:   python src/12_change_photos.py        (about 10 minutes on a laptop)
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageFilter

BATCH = 32          # photos passed through the network at a time
SAVE_EVERY = 10     # batches between two saves of the progress
WEIGHTS = Path("weights/resnet50_ram-a26f946b.pth")
OUT = Path("data")


def blur(crop):
    return np.asarray(Image.fromarray(crop).filter(ImageFilter.GaussianBlur(radius=1.3)))


def dark(crop):
    return (crop * 0.5).astype(np.uint8)


def jpeg(crop):
    file = io.BytesIO()
    Image.fromarray(crop).save(file, "JPEG", quality=20)
    return np.asarray(Image.open(file).convert("RGB"))


CHANGES = {"blur": blur, "dark": dark, "jpeg": jpeg}

if __name__ == "__main__":
    # the network, without its last layer (as in step 3)
    model = torchvision.models.resnet50()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    crops = np.load(OUT / "crops.npy", mmap_mode="r")                       # all photos x 224 x 224 x 3
    aptos = np.flatnonzero(pd.read_csv(OUT / "table.csv")["dataset"] == "aptos")
    n = len(aptos)

    for name, change in CHANGES.items():
        features_file, done_file = OUT / f"features_{name}.npy", OUT / f"features_{name}_done.txt"
        if features_file.exists() and done_file.exists():                   # continue an earlier run
            features = np.load(features_file, mmap_mode="r+")
            done = int(done_file.read_text())
        else:
            features = np.lib.format.open_memmap(features_file, mode="w+", dtype=np.float32, shape=(n, 2048))
            done = 0
        print(f"{name}: {done} of {n} photos already done", flush=True)

        with torch.no_grad():
            for batch_number, start in enumerate(range(done, n, BATCH), start=1):
                rows = aptos[start:start + BATCH]
                batch = np.stack([change(np.array(crops[row])) for row in rows])    # batch x 224 x 224 x 3
                batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255
                features[start:start + BATCH] = model((batch - mean) / std).numpy()
                if batch_number % SAVE_EVERY == 0 or start + BATCH >= n:
                    features.flush()
                    done_file.write_text(str(min(start + BATCH, n)))
                    print(f"{name}: {min(start + BATCH, n)} of {n}", flush=True)

    print(f"\n{n} photos x 2048 features in " + ", ".join(f"data/features_{name}.npy" for name in CHANGES))
