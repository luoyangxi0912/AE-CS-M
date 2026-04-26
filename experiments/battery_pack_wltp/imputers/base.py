from __future__ import annotations

from abc import ABC, abstractmethod


class BaseImputer(ABC):
    """Unified external imputer interface for AE-CS-M."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_source, scaler=None, smoke: bool = False, metadata=None):
        raise NotImplementedError

    @abstractmethod
    def impute(self, X, mask_observed, metadata=None):
        raise NotImplementedError

