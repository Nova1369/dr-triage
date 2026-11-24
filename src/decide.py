"""Decide one photo with the saved rule: no referral, routine referral, urgent referral, or to a grader.

    python src/decide.py datasets/aptos/train_images/000c1434d8d7.png          (one or more photos)

Each photo goes through the whole pipeline of the project:
  1. crop to the retina and resize to 224 x 224        (the function of src/02_crop_photos.py)
  2. 2,048 features from the frozen ResNet-50          (as in src/03_extract_features.py)
  3. three probabilities from the saved model          (models/referral_rule.npz, written by src/17_save_model.py)
  4. prediction set and decision                       (src/conformal.py)

The saved thresholds were set on APTOS photos. For photos of another clinic the 90 % guarantee does not
hold until the thresholds are set again on local photos (steps 7, 8 and 16).
"""
import importlib
import sys
from pathlib import Path

import numpy as np

from conformal import DECISIONS, decide, prediction_sets

MODEL_FILE = Path("models/referral_rule.npz")
WEIGHTS = Path("weights/resnet50_ram-a26f946b.pth")
CLASSES = ["no referral", "routine", "urgent"]


def probabilities(features, model):
    """The three class probabilities of each photo from its 2,048 features, with the saved numbers only."""
    z = (features - model["scaler_mean"]) / model["scaler_scale"]                 # standardise
    components = (z - model["pca_mean"]) @ model["pca_components"].T              # 100 principal components
    scores = components @ model["coefficients"].T + model["intercepts"]           # logistic regression
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))                   # softmax
    return scores / scores.sum(axis=1, keepdims=True)


def resnet_features(crops):
    """224 x 224 x 3 crops (values 0-255) -> 2,048 features each, from the frozen ResNet-50."""
    import torch
    import torchvision
    network = torchvision.models.resnet50()
    network.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    network.fc = torch.nn.Identity()
    network.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        return network((batch - mean) / std).numpy()


if __name__ == "__main__":
    from PIL import Image
    if len(sys.argv) < 2:
        sys.exit("Give one or more photos:   python src/decide.py photo1.png photo2.png")
    crop_retina = importlib.import_module("02_crop_photos").crop_retina            # the cropping of step 2
    model = np.load(MODEL_FILE)
    crops = []
    for path in sys.argv[1:]:
        with Image.open(path) as image:
            crops.append(crop_retina(image))
    p = probabilities(resnet_features(crops), model)
    sets = prediction_sets(p, model["thresholds"])
    for path, p_photo, in_set, decision in zip(sys.argv[1:], p, sets, decide(sets)):
        chosen = [name for name, inside in zip(CLASSES, in_set) if inside]
        print(f"{Path(path).name}:  " + ", ".join(f"P({name}) = {value:.2f}" for name, value in zip(CLASSES, p_photo))
              + f"   set = {{{', '.join(chosen)}}}   ->   {DECISIONS[decision].upper()}")
