import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU
import numpy as np

def dct_filters_tf(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    nf = k**2 if DC else k**2 - 1
    filter_bank = np.zeros((nf, k, k), dtype=np.float32)
    index = 0
    for i in range(k):
        for j in range(k):
            if not DC and i == 0 and j == 0:
                continue
            for x in range(k):
                for y in range(k):
                    filter_bank[index, x, y] = np.cos((np.pi * (x + 0.5) * i) / k) * np.cos((np.pi * (y + 0.5) * j) / k)
            if l1_norm:
                filter_bank[index] /= np.sum(np.abs(filter_bank[index]))
            else:
                ai = np.sqrt(2.0 / k) if i > 0 else np.sqrt(1.0 / k)
                aj = np.sqrt(2.0 / k) if j > 0 else np.sqrt(1.0 / k)
                filter_bank[index] *= ai * aj
            index += 1
    filter_bank = np.tile(filter_bank[np.newaxis, :], (groups, 1, 1, 1))
    return tf.constant(filter_bank, dtype=tf.float32)


class Harm2dTF(tf.keras.layers.Layer):
    def __init__(self, ni, no, kernel_size, stride=1, padding='same', use_bias=True, use_bn=False, level=None, DC=True, groups=1):
        super(Harm2dTF, self).__init__()
        self.dct_filters = dct_filters_tf(k=kernel_size, groups=1, DC=DC)
        self.conv = tf.keras.layers.Conv2D(
            filters=no,
            kernel_size=(1, 1),  # 1x1 convolution to mix the DCT-filtered channels
            strides=stride,
            padding=padding,
            use_bias=use_bias,
            groups=groups
        )
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = tf.nn.depthwise_conv2d(
            input=inputs,
            filter=self.dct_filters,
            strides=[1, 1, 1, 1],  # Corresponds to (batch, height, width, channels)
            padding='SAME'
        )
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        return x
    
def get_model():
    # Use the regular input layer from ResNet50
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(180, 180, 3),
        pooling='avg',
        weights='imagenet'
    )
    base_model.trainable = True

    # Create a new input layer
    input_layer = tf.keras.Input(shape=(180, 180, 3))
    # Pass it through the Harm2dTF layer
    x = Harm2dTF(3, 3, kernel_size=3, stride=1, padding='same', use_bn=True)(input_layer)
    # Then feed it to the base model
    x = base_model(x, training=True)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(150, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model
