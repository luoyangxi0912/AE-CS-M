from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


@tf.keras.utils.register_keras_serializable(package="AECSM", name="gaussian_activation")
def gaussian_activation(x):
    return 1.0 - tf.exp(-tf.square(x))


def _pairwise_time_distances_tf(X, mask):
    X = tf.cast(X, tf.float32)
    mask = tf.cast(mask, tf.float32)
    x_i = X[:, :, tf.newaxis, :]
    x_j = X[:, tf.newaxis, :, :]
    m_i = mask[:, :, tf.newaxis, :]
    m_j = mask[:, tf.newaxis, :, :]
    common = m_i * m_j
    common_counts = tf.reduce_sum(common, axis=-1)
    sq_sum = tf.reduce_sum(tf.square(x_i - x_j) * common, axis=-1)
    inf = tf.constant(float("inf"), dtype=tf.float32)
    distances = tf.where(common_counts > 0.0, tf.sqrt(sq_sum / (common_counts + 1e-8)), inf)
    batch_size = tf.shape(X)[0]
    time_steps = tf.shape(X)[1]
    diag_mask = tf.eye(time_steps, batch_shape=[batch_size], dtype=tf.bool)
    return tf.where(diag_mask, inf, distances)


def _pairwise_feature_distances_tf(X, mask):
    X = tf.cast(X, tf.float32)
    mask = tf.cast(mask, tf.float32)
    X_f = tf.transpose(X, [0, 2, 1])
    M_f = tf.transpose(mask, [0, 2, 1])
    x_i = X_f[:, :, tf.newaxis, :]
    x_j = X_f[:, tf.newaxis, :, :]
    m_i = M_f[:, :, tf.newaxis, :]
    m_j = M_f[:, tf.newaxis, :, :]
    common = m_i * m_j
    common_counts = tf.reduce_sum(common, axis=-1)
    sq_sum = tf.reduce_sum(tf.square(x_i - x_j) * common, axis=-1)
    inf = tf.constant(float("inf"), dtype=tf.float32)
    distances = tf.where(common_counts > 0.0, tf.sqrt(sq_sum / (common_counts + 1e-8)), inf)
    batch_size = tf.shape(X)[0]
    n_features = tf.shape(X)[2]
    diag_mask = tf.eye(n_features, batch_shape=[batch_size], dtype=tf.bool)
    return tf.where(diag_mask, inf, distances)


def _topk_gaussian_weights(distances, k):
    _, indices = tf.nn.top_k(-distances, k=k)
    top_distances = tf.gather(distances, indices, batch_dims=2)
    valid = tf.math.is_finite(top_distances)
    safe_distances = tf.where(valid, top_distances, tf.zeros_like(top_distances))
    valid_count = tf.reduce_sum(tf.cast(valid, tf.float32), axis=-1, keepdims=True)
    sigma = tf.reduce_sum(safe_distances, axis=-1, keepdims=True) / (valid_count + 1e-8)
    sigma = tf.where(valid_count > 0.0, sigma, tf.ones_like(sigma))
    weights = tf.exp(-tf.square(safe_distances) / (tf.square(sigma) + 1e-8))
    weights = tf.where(valid, weights, tf.zeros_like(weights))
    weights = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-8)
    return indices, weights


def compute_spatial_knn_init(X, mask, k=5):
    distances = _pairwise_time_distances_tf(X, mask)
    indices, weights = _topk_gaussian_weights(distances, k)
    neighbor_vals = tf.gather(X, indices, batch_dims=1)
    neighbor_obs = tf.gather(mask, indices, batch_dims=1)
    weighted_vals = neighbor_vals * neighbor_obs * weights[..., tf.newaxis]
    weight_sums = tf.reduce_sum(neighbor_obs * weights[..., tf.newaxis], axis=2)
    filled = tf.reduce_sum(weighted_vals, axis=2) / (weight_sums + 1e-8)
    return tf.where(mask > 0.0, X, tf.where(weight_sums > 0.0, filled, X * mask))


def compute_temporal_knn_init(X, mask, k=5):
    X_f = tf.transpose(X, [0, 2, 1])
    M_f = tf.transpose(mask, [0, 2, 1])
    distances = _pairwise_feature_distances_tf(X, mask)
    indices, weights = _topk_gaussian_weights(distances, k)
    neighbor_vals = tf.gather(X_f, indices, batch_dims=1)
    neighbor_obs = tf.gather(M_f, indices, batch_dims=1)
    weighted_vals = neighbor_vals * neighbor_obs * weights[..., tf.newaxis]
    weight_sums = tf.reduce_sum(neighbor_obs * weights[..., tf.newaxis], axis=2)
    filled = tf.reduce_sum(weighted_vals, axis=2) / (weight_sums + 1e-8)
    x_init_f = tf.where(M_f > 0.0, X_f, tf.where(weight_sums > 0.0, filled, X_f * M_f))
    return tf.transpose(x_init_f, [0, 2, 1])


class Encoder(Model):
    def __init__(self, latent_dim=64, hidden_units=128, dropout_rate=0.1, l2_reg=5e-4, name="encoder"):
        super().__init__(name=name)
        self.noise = layers.GaussianNoise(0.05)
        self.lstm1 = layers.LSTM(hidden_units, return_sequences=True)
        self.lstm2 = layers.LSTM(hidden_units, return_sequences=True)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.latent = layers.Dense(latent_dim, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, x, mask, training=False, pre_filled=False):
        x_input = x if pre_filled else x * mask
        x_input = self.noise(x_input, training=training)
        h = tf.concat([x_input, mask], axis=-1)
        h = self.dropout1(self.norm1(self.lstm1(h, training=training), training=training), training=training)
        h = self.dropout2(self.norm2(self.lstm2(h, training=training), training=training), training=training)
        return self.latent(h)


class Decoder(Model):
    def __init__(self, n_features, hidden_units=128, dropout_rate=0.1, l2_reg=5e-4, name="decoder"):
        super().__init__(name=name)
        self.lstm1 = layers.LSTM(hidden_units, return_sequences=True)
        self.lstm2 = layers.LSTM(hidden_units, return_sequences=True)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.out = layers.Dense(n_features, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, z, training=False):
        h = self.dropout1(self.norm1(self.lstm1(z, training=training), training=training), training=training)
        h = self.dropout2(self.norm2(self.lstm2(h, training=training), training=training), training=training)
        return tf.clip_by_value(self.out(h), -5.0, 5.0)


class GatingNetwork(Model):
    def __init__(self, latent_dim=64, dropout_rate=0.1, l2_reg=5e-4, name="gating_network"):
        super().__init__(name=name)
        self.dense1 = layers.Dense(latent_dim * 2, activation=gaussian_activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dense2 = layers.Dense(latent_dim, activation=gaussian_activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dropout = layers.Dropout(dropout_rate)
        self.alpha = layers.Dense(3, activation="softmax")

    def call(self, z_orig, z_space, z_time, missing_rate, training=False):
        pooled = tf.concat(
            [
                tf.reduce_mean(z_orig, axis=1),
                tf.reduce_mean(z_space, axis=1),
                tf.reduce_mean(z_time, axis=1),
                tf.expand_dims(tf.cast(missing_rate, z_orig.dtype), axis=-1),
            ],
            axis=-1,
        )
        h = self.dense1(pooled)
        h = self.dropout(self.dense2(h), training=training)
        alpha = self.alpha(h)
        alpha_exp = tf.expand_dims(alpha, axis=1)
        z_fused = alpha_exp[:, :, 0:1] * z_orig + alpha_exp[:, :, 1:2] * z_space + alpha_exp[:, :, 2:3] * z_time
        return alpha, z_fused


class AECS(Model):
    def __init__(self, n_features, latent_dim=32, hidden_units=128, k_spatial=5, k_temporal=5, dropout_rate=0.1, l2_reg=5e-4, name="ae_cs"):
        super().__init__(name=name)
        self.k_spatial = k_spatial
        self.k_temporal = k_temporal
        self.encoder_orig = Encoder(latent_dim, hidden_units, dropout_rate, l2_reg, name="encoder_orig")
        self.encoder_space = Encoder(latent_dim, hidden_units, dropout_rate, l2_reg, name="encoder_space")
        self.encoder_time = Encoder(latent_dim, hidden_units, dropout_rate, l2_reg, name="encoder_time")
        self.decoder = Decoder(n_features, hidden_units, dropout_rate, l2_reg)
        self.gating_network = GatingNetwork(latent_dim, dropout_rate, l2_reg)

    def _prepare_inputs(self, x, mask):
        x_zero = x * mask
        x_space = compute_spatial_knn_init(x, mask, self.k_spatial)
        x_time = compute_temporal_knn_init(x, mask, self.k_temporal)
        return x_zero, x_space, x_time

    def call(self, x, mask, training=False, return_all=False):
        missing_rate = 1.0 - tf.reduce_mean(mask, axis=[1, 2])
        x_zero, x_space, x_time = self._prepare_inputs(x, mask)
        z_orig = self.encoder_orig(x_zero, mask, training=training, pre_filled=False)
        z_space = self.encoder_space(x_space, mask, training=training, pre_filled=True)
        z_time = self.encoder_time(x_time, mask, training=training, pre_filled=True)
        alpha, z_fused = self.gating_network(
            tf.nn.l2_normalize(z_orig, axis=-1),
            tf.nn.l2_normalize(z_space, axis=-1),
            tf.nn.l2_normalize(z_time, axis=-1),
            missing_rate,
            training=training,
        )
        x_hat = self.decoder(z_fused, training=training)
        x_filled = x_hat if training else mask * x + (1.0 - mask) * x_hat
        if return_all:
            return {"x_hat": x_hat, "x_filled": x_filled, "z_orig": z_orig, "z_space": z_space, "z_time": z_time, "alpha": alpha}
        return x_filled

    def encode(self, x, mask, training=False):
        return self.encoder_orig(x * mask, mask, training=training, pre_filled=False)
