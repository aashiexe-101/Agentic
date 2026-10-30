import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
tests/test_diagnosis.py — Unit and Integration Test Suite for Agent A2 (Diagnosis).

Tests cover:
1. Model loading & serialization integrity
2. Single-image prediction contract
3. Seven-class probability schema
4. Probability normalization (sum to 1.0 within numerical tolerance)
5. Confidence and argmax invariants
6. Robust error handling on missing and invalid files
7. Determinism and reproducibility across inferences
8. Data loader compatibility (src.data.load_split)
9. End-to-end joint quality + diagnosis pipeline
10. Predict() alias compatibility with diagnose()
"""

import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.common import CLASS_NAMES, DATA_DIR, MODELS_DIR
from src.data import load_split
from src.diagnosis_agent import DiagnosisAgent, diagnose, get_agent, predict
from src.quality_agent import assess_quality


@pytest.fixture(scope="module")
def sample_real_image_path() -> Path:
    """Fixture providing a verified real image path from the dataset."""
    eval_csv = DATA_DIR / "eval" / "metadata.csv"
    if eval_csv.exists():
        df = pd.read_csv(eval_csv)
        first_id = df.iloc[0]["image_id"]
        p = DATA_DIR / "eval" / "images" / f"{first_id}.jpg"
        if p.exists():
            return p
    # Fallback to train
    train_csv = DATA_DIR / "train" / "metadata.csv"
    df = pd.read_csv(train_csv)
    first_id = df.iloc[0]["image_id"]
    return DATA_DIR / "train" / "images" / f"{first_id}.jpg"


class TestModelLoading:
    """Tests for model checkpoint existence and loading."""

    def test_checkpoint_exists(self):
        v2_exists = (MODELS_DIR / "classifier_v2.joblib").exists()
        v1_exists = (MODELS_DIR / "classifier_v1.joblib").exists()
        assert v2_exists or v1_exists, "No trained model checkpoint found in models/"

    def test_agent_initialization(self):
        agent = get_agent()
        assert agent is not None
        assert agent.classifier is not None
        assert agent.feature_extractor is not None
        assert len(agent.classifier.classes_) == len(CLASS_NAMES)


class TestInferenceContract:
    """Tests for public inference interface schema and invariants."""

    def test_single_image_prediction_schema(self, sample_real_image_path):
        result = diagnose(sample_real_image_path)
        assert isinstance(result, dict)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_seven_class_probability_schema(self, sample_real_image_path):
        result = diagnose(sample_real_image_path)
        probs = result["probabilities"]
        assert isinstance(probs, dict)
        assert list(probs.keys()) == CLASS_NAMES
        for cls_name, val in probs.items():
            assert isinstance(val, (float, int))
            assert 0.0 <= val <= 1.0

    def test_probability_normalization(self, sample_real_image_path):
        result = diagnose(sample_real_image_path)
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-3, f"Probabilities do not sum to 1.0: {prob_sum}"

    def test_confidence_and_argmax_invariants(self, sample_real_image_path):
        result = diagnose(sample_real_image_path)
        probs = result["probabilities"]
        pred_class = result["predicted_class"]
        conf = result["confidence"]

        # Pred class must be in CLASS_NAMES
        assert pred_class in CLASS_NAMES

        # Max probability must match confidence
        max_prob = max(probs.values())
        assert math.isclose(conf, max_prob, abs_tol=1e-4)

        # Predicted class must be the argmax
        argmax_class = max(probs, key=probs.get)
        assert pred_class == argmax_class

    def test_predict_alias_compatibility(self, sample_real_image_path):
        res_diag = diagnose(sample_real_image_path)
        res_pred = predict(sample_real_image_path)
        assert res_diag == res_pred


class TestErrorHandling:
    """Tests for defensive error handling on bad inputs."""

    def test_nonexistent_file_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            diagnose("non_existent_image_path_12345.jpg")

    def test_invalid_non_image_file_raises_valueerror(self, tmp_path):
        fake_file = tmp_path / "test.txt"
        fake_file.write_text("not an image file content")
        with pytest.raises(ValueError):
            diagnose(fake_file)


class TestReproducibility:
    """Tests for deterministic outputs on repeated calls."""

    def test_deterministic_predictions(self, sample_real_image_path):
        res1 = diagnose(sample_real_image_path)
        res2 = diagnose(sample_real_image_path)
        assert res1["predicted_class"] == res2["predicted_class"]
        assert math.isclose(res1["confidence"], res2["confidence"], abs_tol=1e-5)
        for c in CLASS_NAMES:
            assert math.isclose(res1["probabilities"][c], res2["probabilities"][c], abs_tol=1e-5)


class TestIntegration:
    """Tests for dataset loader compatibility and end-to-end flow."""

    def test_data_loader_compatibility(self):
        records = load_split("eval")
        assert len(records) == 1942
        img_id, img_path, label = records[0]
        assert img_path.exists()
        assert label in CLASS_NAMES

        # Test inference directly on loaded record
        res = diagnose(img_path)
        assert res["predicted_class"] in CLASS_NAMES

    def test_end_to_end_quality_plus_diagnosis(self, sample_real_image_path):
        q_res = assess_quality(sample_real_image_path)
        assert "status" in q_res
        assert q_res["status"] in ["PASS", "WARN", "FAIL"]

        d_res = diagnose(sample_real_image_path)
        assert d_res["predicted_class"] in CLASS_NAMES
