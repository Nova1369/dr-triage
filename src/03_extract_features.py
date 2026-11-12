"""Step 3. Describe every photo with 2,048 numbers from a pretrained network -> data/features.npy

We do not train a deep network. We take a ResNet-50 that was trained on everyday photos (ImageNet),
remove its last layer (the one that names the 1,000 ImageNet classes), and keep the 2,048 numbers it
computes just before that layer. These numbers describe what is in the photo, and the simple models of
the later steps are trained on them. The network itself is never changed ("frozen").

Each crop from step 2 is scaled to 0-1 and normalised with the ImageNet mean and standard deviation of
each colour channel, as the network expects.

Weights file: weights/resnet50_ram-a26f946b.pth (link in the README).
The script can be stopped at any time: run it again and it continues where it stopped.

Run from the project folder:   python src/03_extract_features.py
"""
from pathlib import Path

import numpy as np
import torch
import torchvision

BATCH = 32          # photos passed through the network at a time
SAVE_EVERY = 10     # batches between two saves of the progress

WEIGHTS = Path("weights/resnet50_ram-a26f946b.pth")
OUT = Path("data")
FEATURES_FILE = OUT / "features.npy"
DONE_FILE = OUT / "features_done.txt"      # how many photos are finished

# the network, without its last layer
model = torchvision.models.resnet50()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
model.fc = torch.nn.Identity()
model.eval()

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

crops = np.load(OUT / "crops.npy", mmap_mode="r")      # n photos x 224 x 224 x 3, values 0-255
n = len(crops)

if FEATURES_FILE.exists() and DONE_FILE.exists():       # continue an earlier run
    features = np.load(FEATURES_FILE, mmap_mode="r+")
    done = int(DONE_FILE.read_text())
else:
    features = np.lib.format.open_memmap(FEATURES_FILE, mode="w+", dtype=np.float32, shape=(n, 2048))
    done = 0
print(f"{done} of {n} photos already done")

with torch.no_grad():
    for batch_number, start in enumerate(range(done, n, BATCH), start=1):
        batch = torch.from_numpy(np.array(crops[start:start + BATCH]))       # batch x 224 x 224 x 3
        batch = batch.permute(0, 3, 1, 2).float() / 255                      # batch x 3 x 224 x 224, 0-1
        features[start:start + BATCH] = model((batch - mean) / std).numpy()
        if batch_number % SAVE_EVERY == 0 or start + BATCH >= n:
            features.flush()
            DONE_FILE.write_text(str(min(start + BATCH, n)))
            print(f"{min(start + BATCH, n)} of {n}", flush=True)

print(f"\n{n} photos x {features.shape[1]} features in {FEATURES_FILE}")
