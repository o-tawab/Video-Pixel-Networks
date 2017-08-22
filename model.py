import tensorflow as tf


class VideoPixelNetworkModel:
    def __init__(self, config):
        self.config = config

    def multiplicative_unit(self, h, dilation_rate):
        with tf.variable_scope('multiplicative_unit'):
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

    def residual_multiplicative_block(self, h, dilation_rate):
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

            h2 = self.multiplicative_unit(h1, dilation_rate)

            h3 = self.multiplicative_unit(h2, dilation_rate)

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
                    x = self.residual_multiplicative_block(x, self.config.encoder_rmb_dilation_scheme[i])
                else:
                    x = self.residual_multiplicative_block(x, 1)

            return x

    def pixel_cnn_decoders(self, h, x):
        with tf.variable_scope('pixel_cnn_decoders'):
            self.residual_multiplicative_block(h, 1)

            tf.concat([x, h], axis=3)
            self.residual_multiplicative_block()


    def conv_lstm(self):
        pass

    def template(self):
        pass

    def build_model(self):
        pass
