import tensorflow as tf
from layers import masked_conv2d


class VideoPixelNetworkModel:
    def __init__(self, config):
        self.config = config

    def multiplicative_unit_without_mask(self, h, dilation_rate):
        with tf.variable_scope('multiplicative_unit_without_mask'):
            g1 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g1'
            )

            g2 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g2'
            )

            g3 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g3'
            )

            u = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='u'
            )

            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)

            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))

            return mu

    def multiplicative_unit_with_mask(self, h, mask_type, masked_channels, in_channels, out_channels):
        with tf.variable_scope('multiplicative_unit_without_mask'):
            g1 = masked_conv2d(
                h,
                in_channels,  # num of channels in input
                out_channels,
                3,
                mask_type,  # A, B or None
                masked_channels,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g1')

            g2 = masked_conv2d(
                h,
                in_channels,  # num of channels in input
                out_channels,
                3,
                mask_type,  # A, B or None
                masked_channels,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g2')

            g3 = masked_conv2d(
                h,
                in_channels,  # num of channels in input
                out_channels,
                3,
                mask_type,  # A, B or None
                masked_channels,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g3')

            u = masked_conv2d(
                h,
                in_channels,  # num of channels in input
                out_channels,
                3,
                mask_type,  # A, B or None
                masked_channels,
                padding='same',
                activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='u')

            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)

            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))

            return mu

    def residual_multiplicative_block_without_mask(self, h, dilation_rate):
        with tf.variable_scope('residual_multiplicative_block'):
            h1 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='h1'
            )

            h2 = self.multiplicative_unit_without_mask(h1, dilation_rate)

            h3 = self.multiplicative_unit_without_mask(h2, dilation_rate)

            h4 = tf.layers.conv2d(
                h3,
                2 * self.config.rmb_c,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='h4'
            )

            rmb = tf.add(h, h4)

            return rmb

    def residual_multiplicative_block_with_mask(self, h, first_block=False, last_block=False):
        with tf.variable_scope('residual_multiplicative_block'):

            if first_block:
                h1 = self.multiplicative_unit_with_mask(
                    h, 'A',
                    self.config.input_shape[2],
                    self.config.input_shape[2] + 2 * self.config.rmb_c,
                    self.config.rmb_c)
            else:
                h1 = tf.layers.conv2d(
                    h,
                    self.config.rmb_c,
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='h1'
                )

            h2 = self.multiplicative_unit_with_mask(
                h1, 'B',
                self.config.rmb_c,
                self.config.rmb_c,
                self.config.rmb_c)

            h3 = self.multiplicative_unit_with_mask(
                h2, 'B',
                self.config.rmb_c,
                self.config.rmb_c,
                self.config.rmb_c)

            if last_block:
                h4 = tf.layers.conv2d(
                    h3,
                    255,
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='h4'
                )
            else:
                h4 = tf.layers.conv2d(
                    h3,
                    2 * self.config.rmb_c,
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='h4'
                )

            rmb = tf.add(h, h4)

            return rmb

    def resolution_preserving_cnn_encoders(self, x):
        with tf.variable_scope('resolution_preserving_cnn_encoders'):
            for i in range(self.config.encoder_rmb_num):
                if self.config.encoder_rmb_dilation:
                    x = self.residual_multiplicative_block_without_mask(x, self.config.encoder_rmb_dilation_scheme[i])
                else:
                    x = self.residual_multiplicative_block_without_mask(x, 1)

            return x

    def pixel_cnn_decoders(self, h, x):
        with tf.variable_scope('pixel_cnn_decoders'):
            h = self.residual_multiplicative_block_without_mask(h, 1)

            h = tf.concat([x, h], axis=3)
            h = self.residual_multiplicative_block_with_mask(h, True)

            for i in range(2, self.config.decoder_rmb_num - 1):
                h = self.residual_multiplicative_block_with_mask(h)

                h = self.residual_multiplicative_block_with_mask(h, last_block=True)

            return h

    def conv_lstm(self):
        pass

    def template(self):
        pass

    def build_model(self):
        pass
