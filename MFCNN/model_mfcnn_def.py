from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Activation, Reshape, MaxPooling2D, AveragePooling2D, UpSampling2D, Dropout, Layer
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Layer
from tensorflow.keras.regularizers import l2

class DoubleConv(Layer):
    def __init__(self, out_channels, mid_channels=None, l2_reg=0.01,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(DoubleConv, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.mid_channels = mid_channels if mid_channels is not None else out_channels
        self.l2_reg = l2_reg
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.conv1 = Conv2D(self.mid_channels, (3, 3), strides=1, padding='same',
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=l2(self.l2_reg))
        self.bn1 = BatchNormalization(axis=-1, momentum=0.99)
        self.act1 = LeakyReLU(negative_slope=0.01)

        self.conv2 = Conv2D(self.out_channels, (3, 3), strides=1, padding='same',
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=l2(self.l2_reg))
        self.bn2 = BatchNormalization(axis=-1, momentum=0.99)
        self.act2 = LeakyReLU(negative_slope=0.01)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x


class FMM(Layer):
    def __init__(self, l2_reg=0.01, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(FMM, self).__init__(**kwargs)
        self.l2_reg = l2_reg
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.stage1_conv = Conv2D(64, (3, 3), strides=2, padding='same',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=l2(self.l2_reg))
        self.stage1_act = LeakyReLU(negative_slope=0.01)
        self.stage1_dc = DoubleConv(128, 96, l2_reg=self.l2_reg,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer)

        self.stage2_max = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.stage2_dc = DoubleConv(256, 192, l2_reg=self.l2_reg,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer)

        self.stage3_max = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.stage3_dc = DoubleConv(512, 256, l2_reg=self.l2_reg,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer)

    def call(self, inputs):
        x = self.stage1_conv(inputs)
        x = self.stage1_act(x)
        stage1_dc = self.stage1_dc(x)

        x = self.stage2_max(stage1_dc)
        stage2_dc = self.stage2_dc(x)

        x = self.stage3_max(stage2_dc)
        stage3_dc = self.stage3_dc(x)

        return stage1_dc, stage2_dc, stage3_dc


class ScaleBlock(Layer):
    def __init__(self, pool_size, l2_reg=0.01, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', **kwargs):
        super(ScaleBlock, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.l2_reg = l2_reg
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.avg_pool = AveragePooling2D(pool_size=(self.pool_size, self.pool_size))
        self.conv1 = Conv2D(256, (1, 1), padding='valid',
                            kernel_regularizer=l2(self.l2_reg),
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer)
        self.relu1 = LeakyReLU(negative_slope=0.01)
        self.upsample = UpSampling2D(size=(self.pool_size, self.pool_size), interpolation='bilinear')
        self.conv2 = Conv2D(256, (3, 3), padding='same',
                            kernel_regularizer=l2(self.l2_reg),
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer)
        self.relu2 = LeakyReLU(negative_slope=0.01)

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        conv1 = self.conv1(avg_pool)
        relu1 = self.relu1(conv1)

        upsample = self.upsample(relu1)
        conv2 = self.conv2(upsample)
        relu2 = self.relu2(conv2)
        return relu2


class PaddingLayer(Layer):
    def __init__(self, **kwargs):
        super(PaddingLayer, self).__init__(**kwargs)

    def call(self, inputs, maxH, maxW):
        diffH = maxH - tf.shape(inputs)[1]
        diffW = maxW - tf.shape(inputs)[2]

        pad_top = diffH // 2
        pad_bottom = diffH - pad_top
        pad_left = diffW // 2
        pad_right = diffW - pad_left

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

        return padded_inputs


class MultiscaleLayer(Layer):
    def __init__(self, l2_reg=0.01, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(MultiscaleLayer, self).__init__(**kwargs)
        self.l2_reg = l2_reg
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.scale_block_16 = ScaleBlock(16, l2_reg=self.l2_reg,
                                         kernel_initializer=self.kernel_initializer,
                                         bias_initializer=self.bias_initializer)
        self.scale_block_8 = ScaleBlock(8, l2_reg=self.l2_reg,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer)
        self.scale_block_4 = ScaleBlock(4, l2_reg=self.l2_reg,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer)
        self.scale_block_2 = ScaleBlock(2, l2_reg=self.l2_reg,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer)
        self.padding_layer = PaddingLayer()

    def call(self, inputs):
        x1 = self.scale_block_16(inputs)
        x2 = self.scale_block_8(inputs)
        x3 = self.scale_block_4(inputs)
        x4 = self.scale_block_2(inputs)

        maxH = tf.reduce_max([tf.shape(x1)[1], tf.shape(x2)[1], tf.shape(x3)[1], tf.shape(x4)[1]])
        maxW = tf.reduce_max([tf.shape(x1)[2], tf.shape(x2)[2], tf.shape(x3)[2], tf.shape(x4)[2]])

        x1 = self.padding_layer(x1, maxH, maxW)
        x2 = self.padding_layer(x2, maxH, maxW)
        x3 = self.padding_layer(x3, maxH, maxW)
        x4 = self.padding_layer(x4, maxH, maxW)

        return tf.concat([x1, x2, x3, x4], axis=-1)

class Up(Layer):
    def __init__(self, out_channels, bn=False, l2_reg=0.01, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', **kwargs):
        super(Up, self).__init__(**kwargs)
        self.bn = bn
        self.out_channels = out_channels
        self.l2_reg = l2_reg
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.conv = Conv2D(self.out_channels, (3, 3), padding='same',
                           kernel_initializer=self.kernel_initializer,
                           bias_initializer=self.bias_initializer,
                           kernel_regularizer=l2(self.l2_reg))
        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')

        if self.bn:
            self.bn_layer = BatchNormalization(axis=-1, momentum=0.99)

        self.activation = LeakyReLU(negative_slope=0.01)

    def call(self, inputs):
        x = self.conv(inputs)
        if self.bn:
            x = self.bn_layer(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x

class OutConv(Layer):
    def __init__(self, out_channels, l2_reg=0.01, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', **kwargs):
        super(OutConv, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.l2_reg = l2_reg
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.conv = Conv2D(self.out_channels, (1, 1),
                           kernel_initializer=self.kernel_initializer,
                           bias_initializer=self.bias_initializer,
                           kernel_regularizer=l2(self.l2_reg))

    def call(self, inputs):
        x = self.conv(inputs)
        return x


class PadByUp(Layer):
    def __init__(self, **kwargs):
        super(PadByUp, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        self._check_shapes(x, y)
        return self.pad_and_concat(x, y)

    def pad_and_concat(self, x, y):
        diffH = tf.shape(x)[1] - tf.shape(y)[1]
        diffW = tf.shape(x)[2] - tf.shape(y)[2]

        pad_top = diffH // 2
        pad_bottom = diffH - pad_top
        pad_left = diffW // 2
        pad_right = diffW - pad_left

        y_padded = tf.pad(y, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        return tf.concat([x, y_padded], axis=-1)

    def _check_shapes(self, x, y):
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)

        tf.debugging.assert_equal(tf.rank(x), 4, "x must be a 4D tensor")
        tf.debugging.assert_equal(tf.rank(y), 4, "y must be a 4D tensor")
        tf.debugging.assert_equal(x_shape[0], y_shape[0], "Batch size of x and y must be the same")



def build_model_mfcnn(num_channels=12, num_classes=1, dropout_p=0.2, l2_reg=0.001):
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

    weight_initializer = RandomNormal(mean=0.0, stddev=0.01)
    bias_initializer = Constant(value=0.1)

    # feature map module
    fmm = FMM(l2_reg=l2_reg, kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
    multiscale_layer = MultiscaleLayer(l2_reg=l2_reg, kernel_initializer=weight_initializer, bias_initializer=bias_initializer)


    x1, x2, x3 = fmm(inputs)

    # multiscale module
    x4 = multiscale_layer(x3)

    # # up-sampling module - Up1
    padbyup_1 = PadByUp()
    pad = padbyup_1((x3, x4))
    up1_layer = Up(512, l2_reg=l2_reg, kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
    up1 = up1_layer(pad)
    #
    # # up-sampling module - Up2
    padbyup_2 = PadByUp()
    pad = padbyup_2((x2, up1))
    up2_layer = Up(256, bn=True, l2_reg=l2_reg, kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
    up2 = up2_layer(pad)

    # # up-sampling module - Up3
    padbyup_3 = PadByUp()
    pad = padbyup_3((x1, up2))
    up3_layer = Up(128, bn=True, l2_reg=l2_reg, kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
    up3 = up3_layer(pad)

    dp = Dropout(rate=dropout_p)(up3)

    # output layer
    out_conv = OutConv(num_classes, l2_reg=l2_reg, kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
    outputs = out_conv(dp)

    if num_classes > 1:
        outputs = Activation('softmax')(outputs)

    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_model_mfcnn(num_channels=12, num_classes=3)
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=optimizer)
    model.summary()
    # model.save('/Users/tosha_008/PycharmProjects/cloudFCN_master/models/try_1.keras')
