from __future__ import annotations

import numpy as np
import tensorflow as tf

from experiments.battery_pack_wltp.imputers.base import BaseImputer
from models.aecs import AECS, generate_augmented_masks, total_loss


DEFAULT_AECS_CONFIG = {
    "latent_dim": 32,
    "hidden_units": 128,
    "dropout_rate": 0.1,
    "batch_size": 8,
    "epochs": 3,
    "learning_rate": 1e-3,
    "l2_reg": 1e-4,
    "lambda_consist": 0.05,
    "lambda_space": 0.05,
    "lambda_time": 0.05,
    "p_drop": 0.2,
    "p_consist": 0.1,
    "shuffle_buffer": 256,
    "window_size": 128,
    "stride": 32,
}


class AECSImputer(BaseImputer):
    name = "aecs"

    def __init__(self, config: dict | None = None):
        cfg = dict(DEFAULT_AECS_CONFIG)
        if config:
            cfg.update(config)
        self.latent_dim = int(cfg["latent_dim"])
        self.hidden_units = int(cfg["hidden_units"])
        self.dropout_rate = float(cfg.get("dropout_rate", 0.1))
        self.batch_size = int(cfg["batch_size"])
        self.epochs = int(cfg["epochs"])
        self.learning_rate = float(cfg["learning_rate"])
        self.l2_reg = float(cfg["l2_reg"])
        self.lambda1 = float(cfg["lambda_consist"])
        self.lambda2 = float(cfg["lambda_space"])
        self.lambda3 = float(cfg["lambda_time"])
        self.p_drop = float(cfg["p_drop"])
        self.p_consist = float(cfg["p_consist"])
        self.shuffle_buffer = int(cfg.get("shuffle_buffer", 256))
        self.window_size = int(cfg.get("window_size", 128))
        self.stride = int(cfg.get("stride", 32))
        self.model = None

    @tf.function(reduce_retracing=True)
    def _infer_step(self, xb, mb):
        return self.model(xb, mb, training=False, return_all=True)["x_hat"]

    def _build_model(self, n_features: int):
        self.model = AECS(
            n_features=n_features,
            latent_dim=self.latent_dim,
            hidden_units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
        )
        return self.model

    def fit(self, train_source, scaler=None, smoke: bool = False, metadata=None):
        metadata = metadata or {}
        if train_source is None:
            raise ValueError("AECSImputer.fit() requires non-None train_source.")

        try:
            from experiments.battery_pack_wltp.dataset import iter_window_batches
        except ModuleNotFoundError as exc:
            if exc.name != "experiments.battery_pack_wltp.dataset":
                raise
            raise RuntimeError(
                "Phase 1B-2 dataset protocol is not available yet. "
                "AECSImputer can be imported and constructed, but fit() "
                "requires experiments.battery_pack_wltp.dataset."
            ) from exc

        try:
            first_batch = next(iter_window_batches(train_source, scaler, self.batch_size, self.window_size, self.stride, smoke=smoke))
        except StopIteration as exc:
            raise ValueError(
                "AECSImputer.fit() generated no training windows. "
                "Check train_source, window_size, stride, and smoke settings."
            ) from exc
        train_X0, _ = first_batch
        self._build_model(train_X0.shape[-1])
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        shuffle_seed = metadata.get("seed", 42)

        for _ in range(self.epochs):
            for xb_np, mb_np in iter_window_batches(
                train_source,
                scaler,
                self.batch_size,
                self.window_size,
                self.stride,
                smoke=smoke,
                shuffle_buffer=self.shuffle_buffer,
                seed=shuffle_seed,
            ):
                xb = tf.convert_to_tensor(xb_np, dtype=tf.float32)
                mb = tf.convert_to_tensor(mb_np, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    bern = tf.cast(tf.random.uniform(tf.shape(mb)) > self.p_drop, tf.float32)
                    corrupted_mask = mb * bern
                    outputs = self.model(xb, corrupted_mask, training=True, return_all=True)
                    x_hat = outputs["x_hat"]
                    z_orig = outputs["z_orig"]
                    z_space = outputs["z_space"]
                    z_time = outputs["z_time"]
                    consistency_masks, consistency_weights = generate_augmented_masks(mb, Q=3, p_drop=self.p_consist)
                    q = tf.shape(consistency_masks)[0]
                    time_steps = tf.shape(xb)[1]
                    n_features = tf.shape(xb)[2]
                    tiled_x = tf.tile(xb[tf.newaxis, ...], [q, 1, 1, 1])
                    tiled_x = tf.reshape(tiled_x, [-1, time_steps, n_features])
                    flat_masks = tf.reshape(consistency_masks, [-1, time_steps, n_features])
                    z_corrupted = self.model.encode(tiled_x, flat_masks, training=True)
                    latent_dim = tf.shape(z_corrupted)[-1]
                    z_corrupted_list = tf.reshape(z_corrupted, [q, tf.shape(xb)[0], time_steps, latent_dim])
                    recon_mask = mb * (1.0 - corrupted_mask)
                    loss, _ = total_loss(
                        xb,
                        x_hat,
                        recon_mask,
                        z_orig,
                        z_space,
                        z_time,
                        z_corrupted_list,
                        consistency_weights,
                        self.lambda1,
                        self.lambda2,
                        self.lambda3,
                    )
                    if self.model.losses:
                        loss = loss + tf.add_n(self.model.losses)
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def impute(self, X, mask_observed, metadata=None):
        del metadata
        if self.model is None:
            raise RuntimeError("AECSImputer.impute() requires a fitted model. Call fit() before impute().")

        try:
            from experiments.battery_pack_wltp.windowing import create_windows, reconstruct_from_windows
        except ModuleNotFoundError as exc:
            if exc.name != "experiments.battery_pack_wltp.windowing":
                raise
            raise RuntimeError(
                "Phase 1B-2 windowing support is not available yet. "
                "AECSImputer can be imported and constructed, but impute() "
                "requires experiments.battery_pack_wltp.windowing."
            ) from exc

        X = np.asarray(X, dtype=np.float32)
        M = np.asarray(mask_observed, dtype=np.float32)
        windows_X, windows_M, starts = create_windows(X, M, self.window_size, self.stride)
        ds = tf.data.Dataset.from_tensor_slices((windows_X, windows_M)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        parts = []
        for xb, mb in ds:
            parts.append(self._infer_step(xb, mb).numpy())
        pred_windows = np.concatenate(parts, axis=0).astype(np.float32)
        pred = reconstruct_from_windows(pred_windows, starts, X.shape[0], X.shape[1], self.window_size)
        return np.where(M == 1.0, X, pred).astype(np.float32)
