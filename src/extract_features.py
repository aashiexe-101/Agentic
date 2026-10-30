"""
Feature extraction pipeline using pretrained EfficientNet-B0.

Extracts 1280-dimensional feature representations from dermoscopy images
and caches them to outputs/ for fast classifier training and validation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.common import CLASS_NAMES, DATA_DIR, IMAGE_SIZE, OUTPUTS_DIR


def get_device() -> torch.device:
    """Detect and return the optimal computing device (CUDA, Apple MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_transforms() -> transforms.Compose:
    """Return standard ImageNet preprocessing transforms matching EfficientNet-B0."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_feature_extractor(device: Optional[torch.device] = None) -> nn.Module:
    """
    Load pretrained EfficientNet-B0 with classification head replaced by Identity.
    Parameters are frozen and placed in evaluation mode.
    """
    if device is None:
        device = get_device()

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)
    return model


def extract_features_from_images(
    image_paths: List[Union[str, Path]],
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Extract 1280-d features from a list of image paths in input order.
    Returns:
        np.ndarray of shape (len(image_paths), 1280)
    """
    if not image_paths:
        return np.empty((0, 1280), dtype=np.float32)

    if device is None:
        device = get_device()

    model = get_feature_extractor(device=device)
    transform = get_transforms()

    all_features = []
    total = len(image_paths)

    for start_idx in range(0, total, batch_size):
        batch_paths = image_paths[start_idx : start_idx + batch_size]
        batch_tensors = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensor = transform(img)
            batch_tensors.append(tensor)

        batch_stack = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            features = model(batch_stack)

        all_features.append(features.cpu().numpy())

        if show_progress:
            processed = min(start_idx + batch_size, total)
            print(f"Processed {processed}/{total} images...")

    return np.concatenate(all_features, axis=0)


def load_dataset_metadata(split: str = "train") -> pd.DataFrame:
    """
    Helper to read dataset metadata for a given split without mutating data.py.
    """
    split_dir = DATA_DIR / split
    csv_path = split_dir / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found for split '{split}' at {csv_path}")

    df = pd.read_csv(csv_path)
    images_dir = split_dir / "images"

    # Add resolved image paths
    df["image_path"] = df["image_id"].apply(lambda x: str(images_dir / f"{x}.jpg"))
    return df


def extract_and_cache_split(
    split: str = "train",
    batch_size: int = 64,
    limit: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and cache features, labels, and image IDs for a split.
    Saves outputs to:
        - outputs/{split}_features.npy
        - outputs/{split}_labels.npy
        - outputs/{split}_image_ids.npy
    """
    df = load_dataset_metadata(split=split)
    if limit is not None:
        df = df.iloc[:limit].copy()

    image_paths = df["image_path"].tolist()
    image_ids = df["image_id"].to_numpy(dtype=str)

    # Map text class labels to canonical indices matching CLASS_NAMES
    label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    labels = np.array([label_to_idx[dx] for dx in df["dx"]], dtype=np.int64)

    print(f"Extracting features for {len(image_paths)} images from split '{split}'...")
    features = extract_features_from_images(
        image_paths=image_paths,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUTS_DIR / f"{split}_features.npy", features)
    np.save(OUTPUTS_DIR / f"{split}_labels.npy", labels)
    np.save(OUTPUTS_DIR / f"{split}_image_ids.npy", image_ids)

    print(f"Saved {split} features to {OUTPUTS_DIR / f'{split}_features.npy'} shape={features.shape}")
    return features, labels, image_ids


def smoke_test(n_samples: int = 5) -> dict:
    """
    Verification test on n_samples training images.
    Confirms shape (n_samples, 1280), finite values, alignment, and device.
    """
    device = get_device()
    df = load_dataset_metadata(split="train").head(n_samples)

    image_paths = df["image_path"].tolist()
    image_ids = df["image_id"].tolist()
    labels = df["dx"].tolist()

    features = extract_features_from_images(
        image_paths=image_paths,
        batch_size=n_samples,
        device=device,
        show_progress=False,
    )

    # Verification checks
    shape_ok = features.shape == (n_samples, 1280)
    finite_ok = bool(np.isfinite(features).all())
    aligned_ok = len(image_ids) == len(labels) == len(features) == n_samples

    return {
        "device": str(device),
        "shape": features.shape,
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "aligned_ok": aligned_ok,
        "image_ids": image_ids,
        "labels": labels,
        "sample_feature_norm": float(np.linalg.norm(features[0])),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature extraction for HAM10000 images")
    parser.add_argument("--smoke", action="store_true", help="Run 5-image smoke test")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to extract (train/val)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction")
    args = parser.parse_args()

    if args.smoke:
        result = smoke_test(5)
        print("Smoke Test Results:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        extract_and_cache_split(split=args.split, batch_size=args.batch_size, show_progress=True)
