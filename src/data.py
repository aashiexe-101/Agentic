"""Agentic DermaGuard — Dataset Loading and Pipeline Module.

Module: src/data.py
Author: Aashi (Data Pipeline Owner)
Project: Agentic DermaGuard Healthcare Hackathon

Responsibilities:
- Project-root-aware path resolution for dataset partitions.
- Metadata ingestion, validation, and schema enforcement using pandas.
- Canonical label constants and bidirectional index-to-label mappings.
- Framework-independent dataset abstraction (`DermaGuardDataset`) loading raw
  images via Pillow without duplicate model-specific preprocessing.
- Robust handling of clinical demographic metadata (e.g. nullable age values).
- Image path safety validation preventing traversal attacks and invalid characters.
- Strict validation of directly injected metadata DataFrames.
- Partition record validation alerting on empty partitions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Optional, Union

# Third-party dependencies declared in requirements.txt
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# 1. Project Path Resolution & Constants
# -----------------------------------------------------------------------------

# Resolve repository root: <project_root>/src/data.py -> <project_root>
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Canonical seven diagnostic classes (strict lowercase, alphabetical order)
CLASS_LABELS: list[str] = [
    "akiec",  # Actinic keratoses and intraepithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc",   # Vascular lesions
]

LABEL_TO_IDX: dict[str, int] = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_LABEL: dict[int, str] = {idx: label for idx, label in enumerate(CLASS_LABELS)}

# Required schema columns in metadata.csv
REQUIRED_COLUMNS: list[str] = [
    "image_id",
    "lesion_id",
    "dx",
    "age",
    "sex",
    "localization",
]

VALID_SPLITS: tuple[str, str] = ("train", "val")

# Legitimate image IDs: alphanumeric, underscores, hyphens (e.g. 'ISIC_0027419')
VALID_IMAGE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _resolve_data_dir(data_dir: Union[str, Path] = "data") -> Path:
    """Resolve data directory relative to project root if not already absolute."""
    path = Path(data_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def validate_image_id(image_id: Any) -> str:
    """Validate and normalize an image identifier.

    Enforces strict security constraints:
    - Rejects non-string inputs (integers, floats, None, NaN, etc.) with ValueError.
    - Rejects empty strings and whitespace.
    - Rejects path separators ('/', '\\'), traversal tokens ('..'), drive colons,
      and null bytes.
    - Strips optional legitimate '.jpg' or '.jpeg' extension (case-insensitive).
    - Enforces alphanumeric, underscore, and hyphen characters only.

    Args:
        image_id: Candidate image identifier string.

    Returns:
        Cleaned base image ID (without extension).

    Raises:
        ValueError: If image_id is not a string, is empty, contains path
            separators/traversal, or contains disallowed characters.
    """
    if not isinstance(image_id, str):
        raise ValueError(
            f"image_id must be a string, got {type(image_id).__name__}."
        )

    clean_id = image_id.strip()
    if not clean_id:
        raise ValueError("image_id cannot be empty or whitespace.")

    # Explicitly reject path traversal and separator tokens
    if "/" in clean_id or "\\" in clean_id:
        raise ValueError(
            f"Invalid image_id '{image_id}': path separators are not permitted."
        )
    if ".." in clean_id:
        raise ValueError(
            f"Invalid image_id '{image_id}': directory traversal tokens ('..') are forbidden."
        )
    if "\x00" in clean_id or ":" in clean_id:
        raise ValueError(
            f"Invalid image_id '{image_id}': contains forbidden control or drive characters."
        )

    # Support legitimate image IDs with or without .jpg / .jpeg extension
    lower_id = clean_id.lower()
    if lower_id.endswith(".jpg"):
        clean_id = clean_id[:-4]
    elif lower_id.endswith(".jpeg"):
        clean_id = clean_id[:-5]

    clean_id = clean_id.strip()
    if not clean_id:
        raise ValueError(
            f"Invalid image_id '{image_id}': base identifier cannot be empty."
        )

    # Validate character set (alphanumeric, underscore, hyphen)
    if not VALID_IMAGE_ID_PATTERN.match(clean_id):
        raise ValueError(
            f"Invalid image_id '{image_id}': contains unexpected filename characters. "
            "Only alphanumeric characters, underscores, and hyphens are permitted."
        )

    return clean_id


# -----------------------------------------------------------------------------
# 2. Path & Metadata Functions
# -----------------------------------------------------------------------------

def get_image_path(
    image_id: str,
    split: str = "train",
    data_dir: Union[str, Path] = "data",
) -> Path:
    """Resolve and verify the file path of a dermatoscopic image.

    Validates image_id to prevent path traversal and verifies that the resolved
    path remains strictly inside the intended partition's images directory.

    Args:
        image_id: Unique image identifier (e.g., 'ISIC_0027419' or 'ISIC_0027419.jpg').
        split: Dataset partition ('train' or 'val').
        data_dir: Root dataset directory path (relative to repo root or absolute).

    Returns:
        pathlib.Path pointing to the verified existing image file.

    Raises:
        ValueError: If split is invalid, image_id is unsafe/malformed, or path escapes images dir.
        FileNotFoundError: If the resolved image file does not exist on disk.
    """
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split '{split}'. Expected one of {VALID_SPLITS}."
        )

    clean_id = validate_image_id(image_id)
    resolved_data_dir = _resolve_data_dir(data_dir)
    images_dir = (resolved_data_dir / split / "images").resolve()
    image_path = (images_dir / f"{clean_id}.jpg").resolve()

    # Ensure resolved path remains strictly within the intended partition directory
    try:
        image_path.relative_to(images_dir)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: image path '{image_path}' escapes the "
            f"intended directory '{images_dir}'."
        )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image '{clean_id}' not found for split '{split}' at: {image_path}"
        )

    return image_path


def load_metadata(
    split: str = "train",
    data_dir: Union[str, Path] = "data",
) -> pd.DataFrame:
    """Load, validate, and enrich the metadata for a given dataset partition.

    Args:
        split: Dataset partition to load ('train' or 'val').
        data_dir: Root dataset directory path.

    Returns:
        pandas.DataFrame enriched with verified 'image_path' objects and validated labels.

    Raises:
        ImportError: If pandas is not installed in the active environment.
        ValueError: If split is invalid, columns are missing, labels are unrecognized,
            or unsafe image_ids are detected.
        FileNotFoundError: If metadata.csv or any referenced image files are missing.
    """
    if pd is None:
        raise ImportError(
            "The 'pandas' package is required to load dataset metadata. "
            "Please ensure it is installed from requirements.txt."
        )

    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split '{split}'. Expected one of {VALID_SPLITS}."
        )

    resolved_data_dir = _resolve_data_dir(data_dir)
    csv_path = resolved_data_dir / split / "metadata.csv"
    images_dir = (resolved_data_dir / split / "images").resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Metadata file for split '{split}' does not exist at: {csv_path}"
        )

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory for split '{split}' does not exist at: {images_dir}"
        )

    # 1. Read metadata CSV
    df = pd.read_csv(csv_path)

    # 2. Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Metadata in '{csv_path}' is missing required columns: {missing_cols}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )

    # 3. Validate and enforce lowercase class labels
    df["dx"] = df["dx"].apply(lambda s: str(s).strip().lower())
    unique_labels = set(df["dx"].unique())
    invalid_labels = unique_labels - set(CLASS_LABELS)
    if invalid_labels:
        raise ValueError(
            f"Unrecognized class label(s) in '{split}' metadata: {sorted(invalid_labels)}. "
            f"Expected canonical classes: {CLASS_LABELS}"
        )

    # 4. Validate image IDs and construct resolved, safe image paths
    clean_ids: list[str] = []
    image_paths: list[Path] = []
    missing_files: list[Path] = []

    for idx, raw_id in enumerate(df["image_id"]):
        try:
            clean_id = validate_image_id(raw_id)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid image_id at row {idx} in '{csv_path}': {e}"
            ) from e

        clean_ids.append(clean_id)
        img_p = (images_dir / f"{clean_id}.jpg").resolve()

        try:
            img_p.relative_to(images_dir)
        except ValueError:
            raise ValueError(
                f"Path traversal detected for image_id '{raw_id}' at row {idx}: "
                f"'{img_p}' escapes '{images_dir}'."
            )

        if not img_p.is_file():
            missing_files.append(img_p)

        image_paths.append(img_p)

    df["image_id"] = clean_ids
    df["image_path"] = image_paths

    # Check for missing files on disk
    if missing_files:
        sample_missing = [str(p) for p in missing_files[:5]]
        raise FileNotFoundError(
            f"Found {len(missing_files)} missing image file(s) referenced in '{csv_path}'. "
            f"Sample missing: {sample_missing}"
        )

    return df


# -----------------------------------------------------------------------------
# 3. DermaGuardDataset (Framework-Independent Abstraction)
# -----------------------------------------------------------------------------

class DermaGuardDataset:
    """Framework-independent dataset for the Agentic DermaGuard project.

    Provides indexed and iterable access to dermatoscopic images (via Pillow)
    and corresponding patient demographic metadata.

    Design Contract:
    - Loads raw images in RGB mode.
    - Preserves native image dimensions (600 × 450 px).
    - Performs NO image resizing, normalization, or feature extraction.
      (Downstream diagnosis agents handle their own model-specific transforms).
    - Normalizes missing clinical 'age' values to None.
    - Fully validates directly injected metadata DataFrames.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: Union[str, Path] = "data",
        metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize the dataset partition.

        Args:
            split: 'train' or 'val'.
            data_dir: Root data directory path.
            metadata: Pre-loaded DataFrame (optional). If None, loads from disk.
        """
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {VALID_SPLITS}."
            )

        self.split: str = split
        self.data_dir: Path = _resolve_data_dir(data_dir)
        self.images_dir: Path = (self.data_dir / split / "images").resolve()

        if metadata is not None:
            self.df: pd.DataFrame = self._validate_and_prepare_metadata(metadata)
        else:
            self.df = load_metadata(split=split, data_dir=self.data_dir).reset_index(drop=True)

    def _validate_and_prepare_metadata(
        self, metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate and normalize a directly injected metadata DataFrame.

        Ensures all REQUIRED_COLUMNS exist, dx labels are valid canonical classes,
        image IDs are safe, and image_path exists and is verified on disk.

        Args:
            metadata: Input DataFrame to validate.

        Returns:
            Sanitized DataFrame with verified image_path and normalized dx.

        Raises:
            TypeError: If metadata is not a pandas DataFrame.
            ValueError: If required columns are missing, labels are invalid, or paths are unsafe.
            FileNotFoundError: If referenced image files do not exist on disk.
        """
        if pd is not None and not isinstance(metadata, pd.DataFrame):
            raise TypeError(
                f"Injected metadata must be a pandas DataFrame, got {type(metadata).__name__}."
            )

        df = metadata.copy().reset_index(drop=True)

        # 1. Validate required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Injected metadata is missing required columns: {missing_cols}. "
                f"Expected schema: {REQUIRED_COLUMNS}"
            )

        # 2. Validate and normalize dx labels against CLASS_LABELS
        df["dx"] = df["dx"].apply(lambda s: str(s).strip().lower())
        unique_labels = set(df["dx"].unique())
        invalid_labels = unique_labels - set(CLASS_LABELS)
        if invalid_labels:
            raise ValueError(
                f"Injected metadata contains unrecognized class label(s): {sorted(invalid_labels)}. "
                f"Expected canonical classes: {CLASS_LABELS}"
            )

        # 3. Ensure image_path exists for every row, or resolve safely using image_id and split
        has_image_path_col = "image_path" in df.columns
        clean_ids: list[str] = []
        image_paths: list[Path] = []
        missing_files: list[Path] = []

        for idx in range(len(df)):
            raw_id = df.at[idx, "image_id"]
            try:
                clean_id = validate_image_id(raw_id)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid image_id at row {idx} in injected metadata: {e}"
                ) from e
            clean_ids.append(clean_id)

            # Determine image path: validate existing or resolve safely
            raw_path = df.at[idx, "image_path"] if has_image_path_col else None
            is_valid_path_entry = (
                raw_path is not None
                and str(raw_path).strip() != ""
                and str(raw_path).lower() != "nan"
            )
            if pd is not None and is_valid_path_entry:
                if hasattr(pd, "isna") and pd.isna(raw_path):
                    is_valid_path_entry = False

            if is_valid_path_entry:
                resolved_p = Path(raw_path).resolve()
                # Verify that injected image_path does not escape the partition directory
                try:
                    resolved_p.relative_to(self.images_dir)
                except ValueError:
                    raise ValueError(
                        f"Unsafe image_path at row {idx}: '{resolved_p}' escapes the "
                        f"intended partition directory '{self.images_dir}'."
                    )
                # Verify that resolved filename matches the validated image_id and expected .jpg filename
                expected_filename = f"{clean_id}.jpg"
                if resolved_p.name.lower() != expected_filename.lower():
                    raise ValueError(
                        f"Mismatched image_path at row {idx}: path filename '{resolved_p.name}' "
                        f"does not match expected image_id filename '{expected_filename}'."
                    )
            else:
                # Safely resolve using clean_id and partition images_dir
                resolved_p = (self.images_dir / f"{clean_id}.jpg").resolve()
                try:
                    resolved_p.relative_to(self.images_dir)
                except ValueError:
                    raise ValueError(
                        f"Path traversal detected for image_id '{clean_id}' at row {idx}: "
                        f"'{resolved_p}' escapes '{self.images_dir}'."
                    )

            if not resolved_p.is_file():
                missing_files.append(resolved_p)

            image_paths.append(resolved_p)

        df["image_id"] = clean_ids
        df["image_path"] = image_paths

        if missing_files:
            sample_missing = [str(p) for p in missing_files[:5]]
            raise FileNotFoundError(
                f"Injected metadata references {len(missing_files)} missing image file(s) on disk. "
                f"Sample missing: {sample_missing}"
            )

        return df

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Fetch a single dataset sample by index.

        Args:
            index: Integer index of the sample.

        Returns:
            Dictionary matching the agreed contract:
            {
                "image_id": str,
                "image_path": Path,
                "image": PIL.Image.Image,
                "label_name": str,
                "label_idx": int,
                "metadata": {
                    "lesion_id": str,
                    "age": float | None,
                    "sex": str,
                    "localization": str,
                }
            }
        """
        if Image is None:
            raise ImportError(
                "The 'Pillow' (PIL) package is required to load image files. "
                "Please ensure it is installed from requirements.txt."
            )

        if index < 0 or index >= len(self.df):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self.df)}."
            )

        row = self.df.iloc[index]

        image_id: str = str(row["image_id"]).strip()
        image_path: Path = Path(row["image_path"])

        # Load image via Pillow and ensure standard 3-channel RGB format
        # Native dimensions (600 x 450) are preserved without resizing
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        label_name: str = str(row["dx"]).strip().lower()
        label_idx: int = LABEL_TO_IDX[label_name]

        # Clinical metadata with robust nullable age handling
        raw_age = row["age"]
        is_missing_age = (
            raw_age is None
            or str(raw_age).strip() == ""
            or str(raw_age).lower() == "nan"
        )
        if pd is not None and not is_missing_age:
            if hasattr(pd, "isna") and pd.isna(raw_age):
                is_missing_age = True

        if is_missing_age:
            age: Optional[float] = None
        else:
            try:
                age = float(raw_age)
            except (ValueError, TypeError):
                age = None

        metadata: dict[str, Any] = {
            "lesion_id": str(row["lesion_id"]).strip(),
            "age": age,
            "sex": str(row["sex"]).strip(),
            "localization": str(row["localization"]).strip(),
        }

        return {
            "image_id": image_id,
            "image_path": image_path,
            "image": image,
            "label_name": label_name,
            "label_idx": label_idx,
            "metadata": metadata,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield dataset samples sequentially."""
        for i in range(len(self)):
            yield self[i]

    def get_class_distribution(self) -> dict[str, int]:
        """Return the count of samples per class label in this partition."""
        counts = self.df["dx"].value_counts().to_dict()
        return {label: int(counts.get(label, 0)) for label in CLASS_LABELS}


# -----------------------------------------------------------------------------
# 4. Sanity-Check CLI Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """Execute validation and diagnostic checks on both dataset partitions."""
    print("=" * 70)
    print("  Agentic DermaGuard — Data Pipeline Sanity Check (`src/data.py`)")
    print("=" * 70)

    # 1. Path Verification
    print(f"\n[Paths]")
    print(f"  Project Root : {PROJECT_ROOT}")
    data_dir = _resolve_data_dir("data")
    print(f"  Data Root    : {data_dir}")

    # Check dependency availability
    print(f"\n[Dependency Check]")
    print(f"  pandas available : {pd is not None}")
    print(f"  Pillow available : {Image is not None}")

    if pd is None or Image is None:
        missing = []
        if pd is None:
            missing.append("pandas")
        if Image is None:
            missing.append("pillow")
        print(f"\n[!] Environment limitation: Missing required packages: {missing}")
        print("    Please install them via: pip install -r requirements.txt")
        return

    # 2. Inspect Partitions
    all_partitions_valid = True

    for split in VALID_SPLITS:
        print(f"\n[{split.upper()} Partition Check]")
        split_dir = data_dir / split
        print(f"  Metadata Path : {split_dir / 'metadata.csv'}")
        print(f"  Images Path   : {split_dir / 'images'}")

        # Instantiate dataset
        dataset = DermaGuardDataset(split=split, data_dir=data_dir)
        total_records = len(dataset)
        print(f"  Total Records : {total_records}")

        # Empty partition check
        if total_records == 0:
            all_partitions_valid = False
            print(
                f"  [FAILURE / WARNING] Partition '{split}' contains 0 records! "
                "Expected a non-empty dataset."
            )
            continue

        # Class Distribution
        dist = dataset.get_class_distribution()
        print(f"  Class Distribution:")
        for label in CLASS_LABELS:
            count = dist[label]
            pct = (count / total_records) * 100 if total_records else 0
            print(f"    - {label:<6} (idx {LABEL_TO_IDX[label]}): {count:>5} ({pct:>5.2f}%)")

        # Verify Image Loading on First 2 Samples
        print(f"\n  [Sample Verification — first 2 records]")
        for i in range(min(2, total_records)):
            sample = dataset[i]
            img = sample["image"]
            print(f"    Sample #{i}:")
            print(f"      image_id   : {sample['image_id']}")
            print(f"      image_path : {sample['image_path'].name}")
            print(f"      dimensions : {img.size} (Width x Height, Mode: {img.mode})")
            print(f"      label      : '{sample['label_name']}' (idx: {sample['label_idx']})")
            print(f"      metadata   : {sample['metadata']}")

    print("\n" + "=" * 70)
    if all_partitions_valid:
        print("  [SUCCESS] All dataset partitions and samples validated successfully.")
    else:
        print("  [FAILURE] Validation failed: one or more partitions were empty or invalid.")
    print("=" * 70)


if __name__ == "__main__":
    main()
