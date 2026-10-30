"""
src/data.py — Data loading module for Agentic DermaGuard.

Author: Aashi (Dataset Onboarding & Liaison) / Ilka Integration
Provides deterministic dataset loading for training and evaluation splits,
strictly enforcing the contract:
    load_split(split) -> List[Tuple[str, Path, str]]
    where records are (image_id, image_path, label)
"""

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from src.common import CLASS_NAMES, DATA_DIR


def get_split_dir(split: str) -> Path:
    """Resolve directory path for a given split ('train' or 'eval'/'val')."""
    split_normalized = split.lower().strip()
    if split_normalized in ("eval", "val"):
        if (DATA_DIR / "eval").exists():
            return DATA_DIR / "eval"
        elif (DATA_DIR / "val").exists():
            return DATA_DIR / "val"
        else:
            raise FileNotFoundError(f"Evaluation dataset directory not found in {DATA_DIR}")
    elif split_normalized == "train":
        train_dir = DATA_DIR / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Training dataset directory not found in {DATA_DIR}")
        return train_dir
    else:
        raise ValueError(f"Unknown split: '{split}'. Expected 'train', 'eval', or 'val'.")


def load_split(split: str = "eval") -> List[Tuple[str, Path, str]]:
    """
    Load dataset split metadata and resolve local image paths.

    Parameters
    ----------
    split : str
        Dataset split to load: 'eval' (or 'val') or 'train'.

    Returns
    -------
    List[Tuple[str, Path, str]]
        List of tuples: (image_id, image_path, label)
        - image_id: Unique ISIC image identifier (e.g., 'ISIC_0026769')
        - image_path: Verified Path object to the .jpg file
        - label: Canonical diagnostic string in CLASS_NAMES
    """
    split_dir = get_split_dir(split)
    metadata_path = split_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Read metadata deterministically
    df = pd.read_csv(metadata_path)

    # Validate essential columns
    required_cols = {"image_id", "dx"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Metadata {metadata_path} missing required columns: {required_cols - set(df.columns)}"
        )

    # Sort deterministically by image_id
    df = df.sort_values(by="image_id", ascending=True).reset_index(drop=True)

    # Resolve image folder
    images_dir = split_dir / "images"
    if not images_dir.exists():
        images_dir = split_dir

    records: List[Tuple[str, Path, str]] = []
    seen_ids = set()

    for _, row in df.iterrows():
        img_id = str(row["image_id"]).strip()
        label = str(row["dx"]).strip()

        # Validate unique IDs
        if img_id in seen_ids:
            raise ValueError(f"Duplicate image_id found in {split}: {img_id}")
        seen_ids.add(img_id)

        # Validate label membership in CLASS_NAMES
        if label not in CLASS_NAMES:
            raise ValueError(
                f"Invalid label '{label}' for image {img_id}. Must be in {CLASS_NAMES}"
            )

        # Resolve image file
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found for {img_id} at {img_path}")

        records.append((img_id, img_path, label))

    return records


if __name__ == "__main__":
    print("[*] Testing src/data.py load_split contract...")
    for split_name in ["train", "eval"]:
        data = load_split(split_name)
        labels = [rec[2] for rec in data]
        label_counts = pd.Series(labels).value_counts().to_dict()
        print(f"[+] Loaded {len(data):,} records for split '{split_name}'")
        print(f"    Class distribution: {label_counts}")
        print(f"    Sample 1: {data[0]}")
        print(f"    Sample 2: {data[1]}")
