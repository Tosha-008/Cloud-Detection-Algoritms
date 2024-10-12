import tensorflow as tf
from tensorflow.keras import layers, Model


class DoubleConv(tf.keras.Model):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = tf.keras.Sequential([
            layers.Conv2D(mid_channels, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(mid_channels, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])

    def call(self, x):
        return self.double_conv(x)


class FMM(tf.keras.Model):
    """Feature map module"""

    def __init__(self, in_channels):
        super(FMM, self).__init__()
        self.stage1 = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
            DoubleConv(64, 128, 96)
        ])
        self.stage2 = tf.keras.Sequential([
            layers.MaxPooling2D(pool_size=2),
            DoubleConv(128, 256, 192)
        ])
        self.stage3 = tf.keras.Sequential([
            layers.MaxPooling2D(pool_size=2),
            DoubleConv(256, 512, 256)
        ])

    def call(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x1, x2, x3


class ScaleBlock(tf.keras.Model):
    """Used in multiscale module"""

    def __init__(self, pool_size):
        super(ScaleBlock, self).__init__()
        self.scale = tf.keras.Sequential([
            layers.AveragePooling2D(pool_size),
            layers.Conv2D(256, kernel_size=1, activation='relu'),
            layers.UpSampling2D(size=pool_size, interpolation='bilinear'),
            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        ])

    def call(self, x):
        return self.scale(x)


class Multiscale(tf.keras.Model):
    """Multiscale module"""

    def __init__(self):
        super(Multiscale, self).__init__()
        self.scale1 = ScaleBlock(16)
        self.scale2 = ScaleBlock(8)
        self.scale3 = ScaleBlock(4)
        self.scale4 = ScaleBlock(2)

    def call(self, x):
        x1 = self.scale1(x)
        x2 = self.scale2(x)
        x3 = self.scale3(x)
        x4 = self.scale4(x)

        # Resolving image size inconsistencies
        maxH = max(x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1])
        maxW = max(x1.shape[2], x2.shape[2], x3.shape[2], x4.shape[2])
        x1 = self._padding(x1, maxH, maxW)
        x2 = self._padding(x2, maxH, maxW)
        x3 = self._padding(x3, maxH, maxW)
        x4 = self._padding(x4, maxH, maxW)
        return tf.concat([x1, x2, x3, x4], axis=-1)

    def _padding(self, x, maxH, maxW):
        diffH, diffW = maxH - tf.shape(x)[1], maxW - tf.shape(x)[2]
        return tf.pad(x, [[0, 0], [diffH // 2, diffH - diffH // 2], [diffW // 2, diffW - diffW // 2], [0, 0]])


class Up(tf.keras.Model):
    """Convolution then upscaling"""

    def __init__(self, in_channels, out_channels, bn=False):
        super(Up, self).__init__()
        if bn:
            self.conv_up = tf.keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.UpSampling2D(size=2, interpolation='bilinear'),
            ])
        else:
            self.conv_up = tf.keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=3, padding='same', activation='relu'),
                layers.UpSampling2D(size=2, interpolation='bilinear'),
            ])

    def call(self, x):
        return self.conv_up(x)


class OutConv(tf.keras.Model):
    """Output convolution layer"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size=1)

    def call(self, x):
        return self.conv(x)


# Function to build the U-Net model
def build_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    fmm = FMM(in_channels=input_shape[2])
    x1, x2, x3 = fmm(inputs)

    # Example of adding multiscale module
    multiscale = Multiscale()
    x = multiscale(x3)

    # Example of adding upsampling block
    up = Up(x.shape[-1], 256)
    x = up(x)

    # Adding output layer
    outputs = OutConv(x.shape[-1], num_classes)(x)

    model = Model(inputs, outputs)
    return model


# Example of using the model
input_shape = (256, 256, 3)  # Example input image shape
num_classes = 10  # Number of classes for segmentation
unet_model = build_unet(input_shape, num_classes)

# Displaying the model architecture
unet_model.summary()
