import os
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Monkey patch Dense, Conv2D, Conv2DTranspose, InputLayer, Flatten, Reshape just in case
layers_to_patch = [
    tf.keras.layers.Dense,
    tf.keras.layers.Conv2D,
    tf.keras.layers.Conv2DTranspose,
    tf.keras.layers.Flatten,
    tf.keras.layers.Reshape,
    tf.keras.layers.InputLayer
]

for layer_cls in layers_to_patch:
    original_init = layer_cls.__init__
    def make_patched_init(orig_init):
        def patched_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            orig_init(self, *args, **kwargs)
        return patched_init
    layer_cls.__init__ = make_patched_init(original_init)

VAE_ENCODER_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "..", "models", "280k model VAE (VAE v2)", "VAE_280K_Outputs", "encoder_280k_final.keras"
))

VAE_DECODER_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "..", "models", "280k model VAE (VAE v2)", "VAE_280K_Outputs", "decoder_280k_final.keras"
))

try:
    print(f"Loading VAE Encoder from {VAE_ENCODER_PATH}")
    vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH, custom_objects={'Sampling': Sampling})
    print("Encoder loaded successfully.")
    
    print(f"Loading VAE Decoder from {VAE_DECODER_PATH}")
    vae_decoder = tf.keras.models.load_model(VAE_DECODER_PATH, custom_objects={'Sampling': Sampling})
    print("Decoder loaded successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
