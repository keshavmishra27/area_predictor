import tensorflow as tf
from tensorflow.keras import layers, Model

TILE_SIZE = 256  # ensure consistency with data_utils

class ChangeDetector(Model):
    def __init__(self, num_filters=32):
        super().__init__()
        # Shared encoder (no pretrained weights for simplicity)
        self.encoder = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(TILE_SIZE, TILE_SIZE, 3)
        )
        # Difference conv
        self.conv_diff = layers.Conv2D(num_filters, 3, padding='same', activation='relu')
        # Decoder: upsampling + conv layers
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(1, 1, activation='sigmoid')
        ])

    def call(self, inputs):
        img1, img2 = inputs
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)
        diff = tf.abs(f1 - f2)
        x = self.conv_diff(diff)
        out = self.decoder(x)
        return out