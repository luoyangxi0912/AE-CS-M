from __future__ import annotations

import tensorflow as tf


def generate_augmented_masks(mask, Q=3, p_drop=0.1):
    mask = tf.cast(mask, tf.float32)
    sample_shape = tf.concat([tf.reshape(tf.cast(Q, tf.int32), [1]), tf.shape(mask)], axis=0)
    bernoulli_mask = tf.cast(tf.random.uniform(sample_shape, dtype=tf.float32) > tf.cast(p_drop, tf.float32), tf.float32)
    corrupted_masks = mask[tf.newaxis, ...] * bernoulli_mask
    diff = tf.abs(corrupted_masks - mask[tf.newaxis, ...])
    l1_distance = tf.reduce_sum(diff, axis=[1, 2, 3])
    n_elements = tf.cast(tf.size(mask), tf.float32)
    sigma_c = tf.sqrt(n_elements) * 0.1
    weights = tf.exp(-l1_distance / (sigma_c**2))
    return corrupted_masks, weights


def reconstruction_loss(x_true, x_pred, recon_mask):
    diff = (x_true - x_pred) * recon_mask
    return tf.reduce_sum(tf.square(diff)) / (tf.reduce_sum(recon_mask) + 1e-8)


def consistency_loss(z_orig, z_corrupted_list, weights):
    z_orig_sg = tf.stop_gradient(z_orig)
    diff = z_corrupted_list - z_orig_sg[tf.newaxis, ...]
    mse = tf.reduce_mean(tf.square(diff), axis=[1, 2, 3])
    weights = tf.cast(weights, tf.float32)
    return tf.reduce_sum(weights * mse) / (tf.reduce_sum(weights) + 1e-8)


def spatial_preservation_loss(z_orig, z_space):
    a = tf.math.l2_normalize(z_orig, axis=-1)
    b = tf.math.l2_normalize(z_space, axis=-1)
    return tf.reduce_mean(tf.reduce_sum(tf.square(a - b), axis=-1))


def temporal_preservation_loss(z_orig, z_time):
    a = tf.math.l2_normalize(z_orig, axis=-1)
    b = tf.math.l2_normalize(z_time, axis=-1)
    return tf.reduce_mean(tf.reduce_sum(tf.square(a - b), axis=-1))


def total_loss(x_true, x_pred, recon_mask, z_orig, z_space, z_time, z_corrupted_list, corruption_weights, lambda1=0.5, lambda2=0.5, lambda3=0.5):
    l_recon = reconstruction_loss(x_true, x_pred, recon_mask)
    l_consist = consistency_loss(z_orig, z_corrupted_list, corruption_weights)
    l_space = spatial_preservation_loss(z_orig, z_space)
    l_time = temporal_preservation_loss(z_orig, z_time)
    total = l_recon + lambda1 * l_consist + lambda2 * l_space + lambda3 * l_time
    return total, {"total": total, "recon": l_recon, "consist": l_consist, "space": l_space, "time": l_time}
