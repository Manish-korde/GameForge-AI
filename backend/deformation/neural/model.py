"""
GameForge AI — Neural Dense Flow Warp Model (Spatial Transformer)
Stage 3 implementation: Differentiable coordinate flow network for 2D character pose deformation.
Learns a continuous backward coordinate displacement field conditioned on joint displacements.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple


@tf.keras.utils.register_keras_serializable()
class BilinearGridWarp(layers.Layer):
    """
    Differentiable 2D bilinear grid sampling layer.
    Warp coordinates: (X_sample, Y_sample) = (X_base + Flow_X, Y_base + Flow_Y)
    Maps normalized coordinates [0, 1] to source image pixels using bilinear interpolation.
    """

    def __init__(self, **kwargs):
        super(BilinearGridWarp, self).__init__(**kwargs)

    def call(self, inputs):
        image, flow = inputs  # image: (B, H, W, C), flow: (B, H, W, 2) in normalized units
        shape = tf.shape(image)
        batch_size, height, width = shape[0], shape[1], shape[2]

        h_f = tf.cast(height, tf.float32)
        w_f = tf.cast(width, tf.float32)

        # 1. Generate base regular normalized grid [0, 1]
        x_lin = tf.linspace(0.0, 1.0, width)
        y_lin = tf.linspace(0.0, 1.0, height)
        grid_x, grid_y = tf.meshgrid(x_lin, y_lin)

        base_grid = tf.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)
        base_grid = tf.expand_dims(base_grid, axis=0)     # (1, H, W, 2)

        # 2. Compute sampling grid: (X_base + delta_x, Y_base + delta_y)
        sample_grid = base_grid + flow  # (B, H, W, 2)
        sample_x = tf.clip_by_value(sample_grid[..., 0] * (w_f - 1.0), 0.0, w_f - 1.0)
        sample_y = tf.clip_by_value(sample_grid[..., 1] * (h_f - 1.0), 0.0, h_f - 1.0)

        # 3. Bilinear pixel sampling
        x0 = tf.floor(sample_x)
        x1 = tf.minimum(x0 + 1.0, w_f - 1.0)
        y0 = tf.floor(sample_y)
        y1 = tf.minimum(y0 + 1.0, h_f - 1.0)

        wa = (x1 - sample_x) * (y1 - sample_y)
        wb = (x1 - sample_x) * (sample_y - y0)
        wc = (sample_x - x0) * (y1 - sample_y)
        wd = (sample_x - x0) * (sample_y - y0)

        wa = tf.expand_dims(wa, axis=-1)
        wb = tf.expand_dims(wb, axis=-1)
        wc = tf.expand_dims(wc, axis=-1)
        wd = tf.expand_dims(wd, axis=-1)

        # Gather pixels across batch
        idx_x0 = tf.cast(x0, tf.int32)
        idx_x1 = tf.cast(x1, tf.int32)
        idx_y0 = tf.cast(y0, tf.int32)
        idx_y1 = tf.cast(y1, tf.int32)

        batch_idx = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
        batch_idx = tf.broadcast_to(batch_idx, [batch_size, height, width])

        def gather_channel(y_idx, x_idx):
            indices = tf.stack([batch_idx, y_idx, x_idx], axis=-1)
            return tf.gather_nd(image, indices)

        Ia = gather_channel(idx_y0, idx_x0)
        Ib = gather_channel(idx_y1, idx_x0)
        Ic = gather_channel(idx_y0, idx_x1)
        Id = gather_channel(idx_y1, idx_x1)

        warped = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return tf.clip_by_value(warped, 0.0, 1.0)


@tf.keras.utils.register_keras_serializable()
class FlowScalingLayer(layers.Layer):
    """Scales raw tanh flow field to maximum normalized displacement."""

    def __init__(self, max_displacement: float = 0.25, **kwargs):
        super(FlowScalingLayer, self).__init__(**kwargs)
        self.max_displacement = max_displacement

    def call(self, inputs):
        return inputs * self.max_displacement

    def get_config(self):
        config = super(FlowScalingLayer, self).get_config()
        config.update({"max_displacement": self.max_displacement})
        return config


def build_flow_stn_model(
    image_shape: Tuple[int, int, int] = (128, 128, 4),
    pose_dim: int = 32,
    max_displacement: float = 0.25
) -> Model:
    """
    Constructs a lightweight Dense Flow Spatial Transformer Model.
    Inputs:
      - source_image: (128, 128, 4) normalized RGBA sprite
      - pose_delta: (32,) normalized joint displacement vector (16 joints * 2)
    Outputs:
      - warped_image: (128, 128, 4) deformed RGBA output
      - flow_field: (128, 128, 2) predicted coordinate displacement field
    """
    src_input = layers.Input(shape=image_shape, name="source_image")
    pose_input = layers.Input(shape=(pose_dim,), name="pose_delta")

    # 1. Condition Pose Delta onto 2D Feature Map
    d1 = layers.Dense(64, activation="relu", name="pose_dense_1")(pose_input)
    d2 = layers.Dense(16 * 16 * 16, activation="relu", name="pose_dense_2")(d1)
    d_spatial = layers.Reshape((16, 16, 16), name="pose_spatial")(d2)
    d_up = layers.UpSampling2D(size=(8, 8), interpolation="bilinear", name="pose_upsample")(d_spatial)  # (128, 128, 16)

    # 2. Extract Spatial Image Features
    c1 = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="img_conv_1")(src_input)
    c2 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="img_conv_2")(c1)

    # 3. Multimodal Concatenation & Flow Estimation
    merged = layers.Concatenate(name="fusion")([c2, d_up])  # (128, 128, 48)

    f1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="flow_conv_1")(merged)
    f2 = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="flow_conv_2")(f1)
    raw_flow = layers.Conv2D(2, (3, 3), padding="same", activation="tanh", name="raw_flow")(f2)

    # Scale flow field using serializable layer
    flow_field = FlowScalingLayer(max_displacement=max_displacement, name="flow_field")(raw_flow)

    # 4. Bilinear Grid Sampling (Warping)
    warped_output = BilinearGridWarp(name="bilinear_warp")([src_input, flow_field])

    model = Model(inputs=[src_input, pose_input], outputs=[warped_output, flow_field], name="FlowSTN")
    return model

