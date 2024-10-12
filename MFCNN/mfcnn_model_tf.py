import tensorflow as tf
from tensorflow.keras import layers, Model
from mcfnn_parts_tf import *

class MFCNN(tf.keras.Model):
    """Full assembly of the parts to form the complete network"""

    def __init__(self, n_channels, n_classes, dropout_p=0.2):
        super(MFCNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Feature map module
        self.fmm = FMM(n_channels)

        # Multiscale module
        self.msm = Multiscale()

        # Up-sampling module
        self.up1 = Up(1536, 512)
        self.up2 = Up(768, 256, bn=True)
        self.up3 = Up(384, 128, bn=True)

        # Output
        self.dp = layers.Dropout(rate=dropout_p)
        self.outc = OutConv(128, n_classes)

    def call(self, x):
        # Feature map module
        x1, x2, x3 = self.fmm(x)

        # Multiscale module
        x4 = self.msm(x3)

        # Up-sampling module
        # Resolving image size inconsistencies
        diffH = x3.shape[1] - x4.shape[1]
        diffW = x3.shape[2] - x4.shape[2]
        x4 = tf.pad(x4, [[0, 0], [diffH // 2, diffH - diffH // 2], [diffW // 2, diffW - diffW // 2], [0, 0]])
        x = self.up1(tf.concat([x3, x4], axis=-1))

        # Resolving image size inconsistencies
        diffH = x2.shape[1] - x.shape[1]
        diffW = x2.shape[2] - x.shape[2]
        x = tf.pad(x, [[0, 0], [diffH // 2, diffH - diffH // 2], [diffW // 2, diffW - diffW // 2], [0, 0]])
        x = self.up2(tf.concat([x2, x], axis=-1))

        # Resolving image size inconsistencies
        diffH = x1.shape[1] - x.shape[1]
        diffW = x1.shape[2] - x.shape[2]
        x = tf.pad(x, [[0, 0], [diffH // 2, diffH - diffH // 2], [diffW // 2, diffW - diffW // 2], [0, 0]])
        x = self.up3(tf.concat([x1, x], axis=-1))

        # Output
        x = self.dp(x)
        return self.outc(x)

# Example of using the model
input_shape = (256, 256, 3)  # Example input image shape
num_classes = 10  # Number of classes for segmentation
mfcnn_model = MFCNN(n_channels=input_shape[2], n_classes=num_classes)

# Compiling the model (if necessary)
mfcnn_model.build(input_shape=(None, *input_shape))
mfcnn_model.summary()  # Displaying the model architecture
