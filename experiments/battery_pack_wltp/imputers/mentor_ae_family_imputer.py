from __future__ import annotations

import numpy as np

from baselines.mentor_ae_family import TRDAE, build_deep_ae
from baselines.mentor_ae_family.imputation_dataset import apply_training_corruption, flatten_windows, reshape_flat_windows
from baselines.mentor_ae_family.module import require_torch
from experiments.battery_pack_wltp.imputers.base import BaseImputer


MENTOR_AE_METHODS = {"deep_ae", "sm_dae", "sdai", "trdae"}

DEFAULT_MENTOR_AE_CONFIG = {
    "batch_size": 4,
    "epochs": 1,
    "learning_rate": 1e-4,
    "hidden_dim": None,
    "latent_dim": None,
    "dropout": 0.0,
    "corruption_rate": 0.2,
    "shuffle_buffer": 64,
    "window_size": 8,
    "stride": 4,
    "device": None,
    "trdae_exact_max_dim": 2048,
}


class MentorAEFamilyImputer(BaseImputer):
    def __init__(self, method: str, config: dict | None = None):
        if method not in MENTOR_AE_METHODS:
            raise ValueError(f"Unsupported mentor AE-family method: {method}")
        cfg = dict(DEFAULT_MENTOR_AE_CONFIG)
        if config:
            cfg.update(config)
        self.name = method
        self.method = method
        self.batch_size = int(cfg["batch_size"])
        self.epochs = int(cfg["epochs"])
        self.learning_rate = float(cfg["learning_rate"])
        self.hidden_dim = cfg["hidden_dim"]
        self.latent_dim = cfg["latent_dim"]
        self.dropout = float(cfg["dropout"])
        self.corruption_rate = float(cfg["corruption_rate"])
        self.shuffle_buffer = int(cfg["shuffle_buffer"])
        self.window_size = int(cfg["window_size"])
        self.stride = int(cfg["stride"])
        self.trdae_exact_max_dim = int(cfg["trdae_exact_max_dim"])
        self.device_spec = cfg["device"]
        self.torch = None
        self.device = None
        self.model = None
        self.input_shape: tuple[int, int] | None = None

    def _ensure_torch(self):
        if self.torch is None or self.device is None:
            self.torch, _ = require_torch()
            self.device = self.torch.device(self.device_spec or ("cuda" if self.torch.cuda.is_available() else "cpu"))
        return self.torch

    def _build_model(self, input_dim: int):
        self._ensure_torch()
        common = {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "device": self.device,
        }
        if self.method == "trdae":
            self.model = TRDAE(input_dim=input_dim, trdae_exact_max_dim=self.trdae_exact_max_dim, **common)
        else:
            self.model = build_deep_ae(self.method, input_dim=input_dim, **common)
        return self.model

    def fit(self, train_source, scaler=None, smoke: bool = False, metadata=None):
        metadata = metadata or {}
        if train_source is None:
            raise ValueError(f"{self.__class__.__name__}.fit() requires non-None train_source.")
        try:
            from experiments.battery_pack_wltp.dataset import iter_window_batches
        except ModuleNotFoundError as exc:
            if exc.name != "experiments.battery_pack_wltp.dataset":
                raise
            raise RuntimeError("Phase 1B dataset protocol is required by mentor AE-family baselines.") from exc

        try:
            first_x, first_mask_observed = next(
                iter_window_batches(train_source, scaler, self.batch_size, self.window_size, self.stride, smoke=smoke)
            )
        except StopIteration as exc:
            raise ValueError(
                f"{self.name}.fit() generated no training windows. "
                "Check train_source, window_size, stride, and smoke settings."
            ) from exc

        first_batch = flatten_windows(first_x, first_mask_observed, self.device)
        self.input_shape = first_x.shape[1:]
        self._build_model(first_batch.values.shape[1])
        self.model.train()
        optimizer = self.torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-10)
        shuffle_seed = metadata.get("seed", 42)

        for _ in range(self.epochs):
            for x_windows, mask_observed_windows in iter_window_batches(
                train_source,
                scaler,
                self.batch_size,
                self.window_size,
                self.stride,
                smoke=smoke,
                shuffle_buffer=self.shuffle_buffer,
                seed=shuffle_seed,
            ):
                batch = flatten_windows(x_windows, mask_observed_windows, self.device)
                corrupted_observed, mask_missing = apply_training_corruption(batch.mask_observed, self.corruption_rate)
                x_input = batch.values * corrupted_observed
                optimizer.zero_grad()
                recon = self.model(x_input, target=batch.values, mask_missing=mask_missing)
                loss = self.model.loss
                if loss is None:
                    raise RuntimeError(f"{self.name} did not produce a training loss.")
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                if self.method in {"sm_dae", "trdae"}:
                    _ = self.model.update_imputation_values(x_input.detach(), recon.detach(), batch.values, mask_missing)
        return self

    def impute(self, X, mask_observed, metadata=None):
        del metadata
        if self.model is None:
            raise RuntimeError(f"{self.name}.impute() requires a fitted model. Call fit() before impute().")
        try:
            from experiments.battery_pack_wltp.windowing import create_windows, reconstruct_from_windows
        except ModuleNotFoundError as exc:
            if exc.name != "experiments.battery_pack_wltp.windowing":
                raise
            raise RuntimeError("Phase 1B windowing support is required by mentor AE-family baselines.") from exc

        X = np.asarray(X, dtype=np.float32)
        mask_observed = np.asarray(mask_observed, dtype=np.float32)
        windows_x, windows_mask_observed, starts = create_windows(X, mask_observed, self.window_size, self.stride)
        self.model.eval()
        pred_windows: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, windows_x.shape[0], self.batch_size):
                end = start + self.batch_size
                batch = flatten_windows(windows_x[start:end], windows_mask_observed[start:end], self.device)
                x_input = batch.values * batch.mask_observed
                recon = self.model(x_input)
                pred_windows.append(reshape_flat_windows(recon, batch.original_shape))
        pred = reconstruct_from_windows(
            np.concatenate(pred_windows, axis=0),
            starts,
            X.shape[0],
            X.shape[1],
            self.window_size,
        )
        return np.where(mask_observed == 1.0, X, pred).astype(np.float32)


class DeepAEImputer(MentorAEFamilyImputer):
    def __init__(self, config: dict | None = None):
        super().__init__("deep_ae", config=config)


class SMDAEImputer(MentorAEFamilyImputer):
    def __init__(self, config: dict | None = None):
        super().__init__("sm_dae", config=config)


class SDAIImputer(MentorAEFamilyImputer):
    def __init__(self, config: dict | None = None):
        super().__init__("sdai", config=config)


class TRDAEImputer(MentorAEFamilyImputer):
    def __init__(self, config: dict | None = None):
        super().__init__("trdae", config=config)
