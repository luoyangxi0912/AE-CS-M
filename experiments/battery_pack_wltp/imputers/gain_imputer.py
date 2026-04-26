from __future__ import annotations

import numpy as np

from baselines.mentor_ae_family.imputation_dataset import apply_training_corruption, flatten_windows, reshape_flat_windows
from baselines.mentor_ae_family.module import require_torch
from baselines.mentor_gain_family import GAIN
from experiments.battery_pack_wltp.imputers.base import BaseImputer


DEFAULT_GAIN_CONFIG = {
    "batch_size": 4,
    "epochs": 1,
    "learning_rate": 1e-4,
    "hidden_dim": None,
    "dropout": 0.0,
    "alpha": 100.0,
    "hint_drop_rate": 0.2,
    "corruption_rate": 0.2,
    "shuffle_buffer": 64,
    "window_size": 8,
    "stride": 4,
    "device": None,
    "d_steps": 1,
    "g_steps": 1,
}


class GAINImputer(BaseImputer):
    name = "gain"

    def __init__(self, config: dict | None = None):
        cfg = dict(DEFAULT_GAIN_CONFIG)
        if config:
            cfg.update(config)
        self.batch_size = int(cfg["batch_size"])
        self.epochs = int(cfg["epochs"])
        self.learning_rate = float(cfg["learning_rate"])
        self.hidden_dim = cfg["hidden_dim"]
        self.dropout = float(cfg["dropout"])
        self.alpha = float(cfg["alpha"])
        self.hint_drop_rate = float(cfg["hint_drop_rate"])
        self.corruption_rate = float(cfg["corruption_rate"])
        self.shuffle_buffer = int(cfg["shuffle_buffer"])
        self.window_size = int(cfg["window_size"])
        self.stride = int(cfg["stride"])
        self.device_spec = cfg["device"]
        self.d_steps = int(cfg["d_steps"])
        self.g_steps = int(cfg["g_steps"])
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
        self.model = GAIN(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            alpha=self.alpha,
            hint_drop_rate=self.hint_drop_rate,
            device=self.device,
        )
        return self.model

    def fit(self, train_source, scaler=None, smoke: bool = False, metadata=None):
        metadata = metadata or {}
        if train_source is None:
            raise ValueError("GAINImputer.fit() requires non-None train_source.")
        try:
            from experiments.battery_pack_wltp.dataset import iter_window_batches
        except ModuleNotFoundError as exc:
            if exc.name != "experiments.battery_pack_wltp.dataset":
                raise
            raise RuntimeError("Phase 1 dataset protocol is required by GAIN.") from exc

        try:
            first_x, first_mask_observed = next(
                iter_window_batches(train_source, scaler, self.batch_size, self.window_size, self.stride, smoke=smoke)
            )
        except StopIteration as exc:
            raise ValueError(
                "gain.fit() generated no training windows. Check train_source, window_size, stride, and smoke settings."
            ) from exc

        self._ensure_torch()
        seed = int(metadata.get("seed", 42))
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

        first_batch = flatten_windows(first_x, first_mask_observed, self.device)
        self.input_shape = first_x.shape[1:]
        self._build_model(first_batch.values.shape[1])
        self.model.train()
        g_optimizer = self.torch.optim.RMSprop(self.model.generator_parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-10)
        d_optimizer = self.torch.optim.RMSprop(self.model.discriminator_parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-10)
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
                corrupted_observed, _ = apply_training_corruption(batch.mask_observed, self.corruption_rate)

                for _ in range(max(1, self.d_steps)):
                    d_optimizer.zero_grad()
                    d_loss = self.model.discriminator_loss(batch.values, corrupted_observed)
                    d_loss.backward()
                    self.torch.nn.utils.clip_grad_norm_(self.model.discriminator_parameters(), 1.0)
                    d_optimizer.step()

                for _ in range(max(1, self.g_steps)):
                    g_optimizer.zero_grad()
                    g_loss = self.model.generator_loss(batch.values, corrupted_observed)
                    g_loss.backward()
                    self.torch.nn.utils.clip_grad_norm_(self.model.generator_parameters(), 1.0)
                    g_optimizer.step()
        return self

    def impute(self, X, mask_observed, metadata=None):
        metadata = metadata or {}
        if self.model is None:
            raise RuntimeError("gain.impute() requires a fitted model. Call fit() before impute().")
        try:
            from experiments.battery_pack_wltp.windowing import create_windows, reconstruct_from_windows
        except ModuleNotFoundError as exc:
            if exc.name != "experiments.battery_pack_wltp.windowing":
                raise
            raise RuntimeError("Phase 1 windowing support is required by GAIN.") from exc

        self._ensure_torch()
        if "seed" in metadata:
            self.torch.manual_seed(int(metadata["seed"]))
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(int(metadata["seed"]))

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
                recon = self.model.complete(x_input, batch.mask_observed)
                pred_windows.append(reshape_flat_windows(recon, batch.original_shape))
        pred = reconstruct_from_windows(
            np.concatenate(pred_windows, axis=0),
            starts,
            X.shape[0],
            X.shape[1],
            self.window_size,
        )
        return np.where(mask_observed == 1.0, X, pred).astype(np.float32)
