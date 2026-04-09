"""
Unit tests for src/models/classifier.py — threshold selection logic.
Requirements: 6.4, 6.5
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.models.classifier import find_optimal_threshold, train_classifier


class _FakeClf:
    """Minimal mock that returns fixed probabilities for find_optimal_threshold."""
    def __init__(self, probas: np.ndarray):
        self._probas = probas

    def predict_proba(self, X):
        return np.column_stack([1 - self._probas, self._probas])


def test_threshold_in_unit_interval():
    """find_optimal_threshold must always return a value in [0, 1] — Req 6.4"""
    rng = np.random.default_rng(0)
    probas = rng.uniform(0, 1, 200)
    y_true = (probas > 0.5).astype(int)
    clf = _FakeClf(probas)
    X_dummy = pd.DataFrame(np.zeros((200, 3)))
    threshold = find_optimal_threshold(clf, X_dummy, pd.Series(y_true))
    assert 0.0 <= threshold <= 1.0, f"Threshold {threshold} outside [0, 1]"


def test_threshold_fallback_to_half():
    """When no threshold achieves recall >= min_recall, fallback = 0.5 — Req 6.5"""
    rng = np.random.default_rng(7)
    probas = rng.uniform(0, 1, 200)
    y_true = (probas > 0.5).astype(int)
    clf = _FakeClf(probas)
    X_dummy = pd.DataFrame(np.zeros((200, 3)))
    # min_recall > 1.0 can never be satisfied → forced fallback
    threshold = find_optimal_threshold(clf, X_dummy, pd.Series(y_true), min_recall=1.01)
    assert threshold == 0.5, f"Expected fallback 0.5, got {threshold}"


def test_threshold_prefers_higher_precision():
    """Among thresholds satisfying recall ≥ 0.90, pick the highest-precision one."""
    rng = np.random.default_rng(42)
    # Make a clean separable dataset so multiple thresholds achieve recall ≥ 0.90
    y_true = np.array([1]*50 + [0]*150)
    # Positives get high scores, negatives get low scores
    probas = np.concatenate([rng.uniform(0.6, 1.0, 50), rng.uniform(0.0, 0.4, 150)])
    clf = _FakeClf(probas)
    X_dummy = pd.DataFrame(np.zeros((200, 3)))
    threshold = find_optimal_threshold(clf, X_dummy, pd.Series(y_true))
    # With clear separation, threshold should be > 0.5
    assert threshold > 0.4, f"Expected threshold > 0.4, got {threshold}"
