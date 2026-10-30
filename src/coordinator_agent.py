"""
src/coordinator_agent.py — Agent A6: Multi-Agent Triage Coordinator (Compatibility Wrapper).

Provides drop-in compatibility for callers referencing either `coordinator_agent.py`
or `coordinator.py`. All logic, constants, contracts, and submission engine utilities
are maintained in `src/coordinator.py`.

Author: Palak (Integration Owner & Agent A6 Coordinator)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

from src.coordinator import (
    ALLOWED_DECISIONS,
    CLASS_NAMES,
    POLICY_VERSION,
    SUBMISSION_COLUMNS,
    coordinate_triage,
    generate_submission_csv,
    validate_submission_csv,
    coordinate,
    triage_case,
)

__all__ = [
    "ALLOWED_DECISIONS",
    "CLASS_NAMES",
    "POLICY_VERSION",
    "SUBMISSION_COLUMNS",
    "coordinate_triage",
    "coordinate",
    "triage_case",
    "generate_submission_csv",
    "validate_submission_csv",
]
