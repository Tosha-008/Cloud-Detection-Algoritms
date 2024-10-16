from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Activation, Reshape, MaxPooling2D, AveragePooling2D, UpSampling2D, Dropout, Layer
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta


def DoubleConv(inputs, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels

    x = Conv2D(mid_channels, (3, 3), strides=1, padding='same', kernel_initializer='glorot_uniform')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(out_channels, (3, 3), strides=1, padding='same', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def FMM(inputs):
    stage1_conv = Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer='glorot_uniform')(inputs)
    stage1_act = Activation('relu')(stage1_conv)
    stage1_dc = DoubleConv(stage1_act, 128, 96)

    stage_2_max = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(stage1_dc)
    stage_2_dc = DoubleConv(stage_2_max, 256, 192)

    stage_3_max = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(stage_2_dc)
    stage_3_dc = DoubleConv(stage_3_max, 512, 256)

    return stage1_dc, stage_2_dc, stage_3_dc


# def ScaleBlock(inputs, pool_size):
#     avg_pool = AveragePooling2D(pool_size=(pool_size, pool_size))(inputs)
#     conv1 = Conv2D(256, (1, 1), padding='valid')(avg_pool)
#     relu1 = Activation('relu')(conv1)
#
#     upsample = UpSampling2D(size=(pool_size, pool_size), interpolation='bilinear')(relu1)
#     conv2 = Conv2D(256, (3, 3), padding='same')(upsample)
#     relu2 = Activation('relu')(conv2)
#     return relu2


class ScaleBlock(Layer):
    def __init__(self, pool_size):
        super(ScaleBlock, self).__init__()
        self.avg_pool = AveragePooling2D(pool_size=(pool_size, pool_size))
        self.conv1 = Conv2D(256, (1, 1), padding='valid')
        self.relu1 = Activation('relu')
        self.upsample = UpSampling2D(size=(pool_size, pool_size), interpolation='bilinear')
        self.conv2 = Conv2D(256, (3, 3), padding='same')
        self.relu2 = Activation('relu')

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        conv1 = self.conv1(avg_pool)
        relu1 = self.relu1(conv1)

        upsample = self.upsample(relu1)
        conv2 = self.conv2(upsample)
        relu2 = self.relu2(conv2)
        return relu2


class PaddingLayer(Layer):
    def call(self, inputs, maxH, maxW):
        return _padding(inputs, maxH, maxW)


def _padding(inputs, maxH, maxW):
    diffH = maxH - tf.shape(inputs)[1]
    diffW = maxW - tf.shape(inputs)[2]

    pad_top = diffH // 2
    pad_bottom = diffH - pad_top
    pad_left = diffW // 2
    pad_right = diffW - pad_left

    return tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])


class MultiscaleLayer(Layer):
    def __init__(self):
        super(MultiscaleLayer, self).__init__()

    def call(self, inputs):
        x1 = ScaleBlock(16)(inputs)
        x2 = ScaleBlock(8)(inputs)
        x3 = ScaleBlock(4)(inputs)
        x4 = ScaleBlock(2)(inputs)

        maxH = tf.reduce_max([tf.shape(x1)[1], tf.shape(x2)[1], tf.shape(x3)[1], tf.shape(x4)[1]])
        maxW = tf.reduce_max([tf.shape(x1)[2], tf.shape(x2)[2], tf.shape(x3)[2], tf.shape(x4)[2]])

        # padding_layer = PaddingLayer()
        x1 = _padding(x1, maxH, maxW)
        x2 = _padding(x2, maxH, maxW)
        x3 = _padding(x3, maxH, maxW)
        x4 = _padding(x4, maxH, maxW)

        return tf.concat([x1, x2, x3, x4], axis=-1)


# def Multiscale(inputs):
#     x1 = ScaleBlock(inputs, 16)
#     x2 = ScaleBlock(inputs, 8)
#     x3 = ScaleBlock(inputs, 4)
#     x4 = ScaleBlock(inputs, 2)
#
#     maxH =
#     maxW =
#
#     x1 = _padding(x1, maxH, maxW)
#     x2 = _padding(x2, maxH, maxW)
#     x3 = _padding(x3, maxH, maxW)
#     x4 = _padding(x4, maxH, maxW)
#     return tf.concat([x1, x2, x3, x4], axis=-1)


def Up(inputs, out_channels, bn=False):
    if bn:
        x = Conv2D(out_channels, (3, 3), padding='same', kernel_initializer='glorot_uniform')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    else:
        x = Conv2D(out_channels, (3, 3), padding='same', kernel_initializer='glorot_uniform')(inputs)
        x = Activation('relu')(x)
        upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    return upsample


def OutConv(inputs, out_channels):
    x = Conv2D(out_channels, (1, 1), kernel_initializer='glorot_uniform')(inputs)
    return x


class Pad_by_up(Layer):
    def call(self, x, y):
        return how_to_pad(x, y)


def how_to_pad(x, y):
    diffH = tf.shape(x)[1] - tf.shape(y)[1]
    diffW = tf.shape(x)[2] - tf.shape(y)[2]

    pad_top = diffH // 2
    pad_bottom = diffH - pad_top
    pad_left = diffW // 2
    pad_right = diffW - pad_left

    y = tf.pad(y, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return tf.concat([x, y], axis=-1)


def build_model_mfcnn(num_channels=12, num_classes=1, dropout_p=0.2):
    """
    This function build_model_mfcnn builds a multiscale fully convolutional neural network (MFCNN) designed for image
    segmentation or similar tasks involving spatial data. The model combines feature extraction, multiscale processing,
    and upsampling, incorporating padding to ensure the feature maps are spatially aligned at
    different stages of the network.

    Parameters
    ----------
    num_channels (int, default=12): The number of input channels.
    This could represent the number of bands or features in the input image.

    num_classes (int, default=1): The number of output classes. For binary segmentation, num_classes=1,
    while for multiclass segmentation, it would be greater than 1.

    dropout_p (float, default=0.2): Dropout probability used to regularize the network.
    This value defines the fraction of units to drop in the final layers.

    Returns OutConv

    """

    inputs = Input(shape=(None, None, num_channels))

    # feature map module
    x1, x2, x3 = FMM(inputs)

    # multiscale module
    multiscale_layer = MultiscaleLayer()
    x4 = multiscale_layer(x3)

    # x4 = Multiscale(x3)

    # up-sampling module - Up1

    # resolving image size inconsistencies
    pad_ = Pad_by_up()
    pad = pad_(x3, x4)
    up1 = Up(pad, 512)

    # up-sampling module - Up2
    pad = pad_(x2, up1)
    up2 = Up(pad, 256, bn=True)

    # up-sampling module - Up3
    pad = pad_(x1, up2)
    up3 = Up(pad, 128, bn=True)

    dp = Dropout(rate=dropout_p)(up3)

    outc = OutConv(dp, num_classes)
    return Model(inputs=inputs, outputs=outc)


if __name__ == "__main__":
    model = build_model_mfcnn(num_channels=12, num_classes=2)
    optimizer = Adadelta()

    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=optimizer)
    model.summary()
