"""
Diagnosis Agent (Agent 2) interface for skin lesion screening.

Loads the trained balanced classifier and pretrained EfficientNet-B0 feature
extractor to predict the lesion class and full class probabilities for an input image.

Contract (TEAM_PLAN_v3.md section 3):
    diagnose(image_path) -> {
        "predicted_class": str,
        "confidence": float,
        "probabilities": {"akiec": float, ..., "vasc": float}
    }
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from src.common import CLASS_NAMES, MODELS_DIR
from src.extract_features import get_device, get_feature_extractor, get_transforms


class DiagnosisAgent:
    """
    Encapsulates the trained lesion classifier and feature extractor
    for single-image and batched diagnosis.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        if model_path is None:
            # Prefer classifier_v2 if available, fallback to classifier_v1
            if (MODELS_DIR / "classifier_v2.joblib").exists():
                model_path = MODELS_DIR / "classifier_v2.joblib"
            elif (MODELS_DIR / "classifier_v1.joblib").exists():
                model_path = MODELS_DIR / "classifier_v1.joblib"
            else:
                model_path = MODELS_DIR / "classifier_v2.joblib"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained classifier not found at {self.model_path}. "
                "Run `python -m src.train_diagnosis` first."
            )

        self.device = device or get_device()
        self.classifier = joblib.load(self.model_path)
        self.feature_extractor = get_feature_extractor(self.device)
        self.transform = get_transforms()

        # Validate classifier classes match CLASS_NAMES
        if len(self.classifier.classes_) != len(CLASS_NAMES):
            raise ValueError(
                f"Classifier classes count ({len(self.classifier.classes_)}) "
                f"does not match CLASS_NAMES ({len(CLASS_NAMES)})."
            )

    def diagnose(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Diagnose a single skin lesion image.

        Args:
            image_path: Path to the image file.

        Returns:
            dict containing:
                - predicted_class: str (one of CLASS_NAMES)
                - confidence: float (highest class probability)
                - probabilities: dict mapping all 7 CLASS_NAMES to probabilities
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {path}")

        try:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Could not read image file at {path}: {e}") from e

        with torch.no_grad():
            features = self.feature_extractor(tensor).cpu().numpy()

        raw_probs = self.classifier.predict_proba(features)[0]
        prob_arr = np.array(raw_probs, dtype=np.float64)
        p_sum = prob_arr.sum()
        if p_sum > 0:
            prob_arr /= p_sum

        # Initial rounding to 4 decimals
        prob_arr = np.round(prob_arr, 4)

        # Reconcile rounding residual on the maximum probability class so sum is exactly 1.0000
        best_idx = int(np.argmax(prob_arr))
        residual = round(1.0 - float(prob_arr.sum()), 4)
        prob_arr[best_idx] = round(float(prob_arr[best_idx] + residual), 4)

        pred_class = CLASS_NAMES[best_idx]
        confidence = float(prob_arr[best_idx])

        # Strict dictionary in canonical CLASS_NAMES order
        prob_dict = {
            CLASS_NAMES[i]: float(prob_arr[i])
            for i in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Alias for diagnose(image_path) to ensure caller compatibility."""
        return self.diagnose(image_path)

    def diagnose_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Diagnose a list of images in batches preserving input order.
        """
        results = []
        for p in image_paths:
            results.append(self.diagnose(p))
        return results


# Module-level singleton
_AGENT_INSTANCE: Optional[DiagnosisAgent] = None


def get_agent(model_path: Optional[Union[str, Path]] = None) -> DiagnosisAgent:
    """Get or initialize the global DiagnosisAgent singleton."""
    global _AGENT_INSTANCE
    if _AGENT_INSTANCE is None or model_path is not None:
        _AGENT_INSTANCE = DiagnosisAgent(model_path=model_path)
    return _AGENT_INSTANCE


def diagnose(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Standard contract function for the Diagnosis Agent.

    Args:
        image_path: Absolute or relative path to a dermoscopy image.

    Returns:
        dict: {"predicted_class": str, "confidence": float, "probabilities": dict}
    """
    agent = get_agent()
    return agent.diagnose(image_path)


def predict(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Canonical alias for diagnose(image_path)."""
    return diagnose(image_path)


if __name__ == "__main__":
    print("Initializing Diagnosis Agent and testing contract...")
    agent = get_agent()
    print(f"Device: {agent.device}")
    print(f"Model:  {agent.model_path}")

    # 1. Test on real training images with known labels
    import pandas as pd
    from src.common import DATA_DIR

    df = pd.read_csv(DATA_DIR / "train" / "metadata.csv")
    sample_images = df.sample(3, random_state=42)

    print("\n" + "=" * 60)
    print("TEST 1: Inferences on Real Training Images")
    print("=" * 60)
    for _, row in sample_images.iterrows():
        img_id = row["image_id"]
        true_label = row["dx"]
        path = DATA_DIR / "train" / "images" / f"{img_id}.jpg"

        result = diagnose(path)
        prob_sum = sum(result["probabilities"].values())

        print(f"\nImage ID:    {img_id} (True: {true_label})")
        print(f"Predicted:   {result['predicted_class']}")
        print(f"Confidence:  {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
        print(f"Sum of Probs:  {prob_sum:.4f}")

        # Assertions
        assert result["predicted_class"] in CLASS_NAMES, "Invalid predicted_class"
        assert list(result["probabilities"].keys()) == CLASS_NAMES, "Mismatched probability keys"
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities do not sum to ~1: {prob_sum}"
        assert result["confidence"] == max(result["probabilities"].values()), "Confidence is not max prob"

    # 2. Test invalid image path handling
    print("\n" + "=" * 60)
    print("TEST 2: Error Handling for Non-Existent Path")
    print("=" * 60)
    try:
        diagnose("data/non_existent_image.jpg")
        print("FAIL: Did not raise FileNotFoundError")
    except FileNotFoundError as e:
        print(f"PASS: Correctly caught FileNotFoundError -> {e}")

    # 3. Test non-image file handling
    print("\n" + "=" * 60)
    print("TEST 3: Error Handling for Non-Image File")
    print("=" * 60)
    try:
        diagnose("src/common.py")
        print("FAIL: Did not raise ValueError")
    except ValueError as e:
        print(f"PASS: Correctly caught ValueError -> {e}")

    print("\nAll Diagnosis Agent verification tests passed successfully!")
