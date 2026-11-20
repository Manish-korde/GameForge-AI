import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

LATENT_DIM = 128

def build_generator(latent_dim=LATENT_DIM):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=42)
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,), name="latent_input"),
        layers.Dense(4 * 4 * 512, use_bias=False, kernel_initializer=init, name="dense_proj"),
        layers.BatchNormalization(name="bn_proj"),
        layers.LeakyReLU(0.2, name="lrelu_proj"),
        layers.Reshape((4, 4, 512), name="reshape"),
        
        # Block 1: 4x4 -> 8x8
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_1"),
        layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.LeakyReLU(0.2, name="lrelu_1"),
        
        # Block 2: 8x8 -> 16x16
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_2"),
        layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.LeakyReLU(0.2, name="lrelu_2"),
        
        # Block 3: 16x16 -> 32x32
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_3"),
        layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init, name="conv_3"),
        layers.BatchNormalization(name="bn_3"),
        layers.LeakyReLU(0.2, name="lrelu_3"),
        
        # Block 4: 32x32 -> 64x64 RGBA
        layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_4"),
        layers.Conv2D(4, kernel_size=3, padding="same", activation="tanh", kernel_initializer=init, name="conv_out")
    ], name="dcgan_generator")
    return model

print("Building models...")
gen_pilot = build_generator()
pilot_weights = r'C:\Users\manis\Downloads\dcgan_frontfacing_centroid_pilot_results\generator_weights_final.weights.h5'
gen_pilot.load_weights(pilot_weights)
print("Pilot 1 weights loaded successfully! Param count:", gen_pilot.count_params())

gen_base = build_generator()
base_weights = r'C:\Users\manis\Downloads\dcgan_frontfacing_46k_50ep_results\generator_weights_final.weights.h5'
gen_base.load_weights(base_weights)
print("Baseline weights loaded successfully! Param count:", gen_base.count_params())
