import tensorflow as tf
from layers import masked_conv2d, BasicConvLSTMCell


class VideoPixelNetworkModel:
    def __init__(self, config):
        self.config = config

        with tf.variable_scope('inputs'):
            self.sequences = tf.placeholder(tf.float32,
                                            shape=[config.batch_size, config.truncated_steps + 1] + config.input_shape,
                                            name='sequences')
            self.inference_prev_frame = tf.placeholder(tf.float32,
                                                       shape=[config.batch_size] + config.input_shape,
                                                       name='inference_prev_frame')
            self.inference_encoder_state = tf.placeholder(tf.float32,
                                                          shape=[config.batch_size, config.input_shape[0],
                                                                 config.input_shape[1], config.conv_lstm_filters],
                                                          name='inference_encoder_state')
            self.inference_current_frame = tf.placeholder(tf.float32,
                                                          shape=[config.batch_size] + config.input_shape,
                                                          name='inference_current_frame')
            self.initial_lstm_state = tf.placeholder(tf.float32,
                                                     shape=[2, config.batch_size, config.input_shape[0],
                                                            config.input_shape[1], config.conv_lstm_filters],
                                                     name='initial_lstm_state')

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

    def conv_lstm(self, x, h):
        with tf.variable_scope('Conv_LSTM'):
            convlstm_cell = BasicConvLSTMCell(
                [self.config.input_shape[0], self.config.input_shape[1]]
                [3, 3],
                self.config.conv_lstm_filters,
                initializer=tf.contrib.layers.xavier_initializer())

            output, state = convlstm_cell(x, h, 'conv_lstm')

            return output, state

    def encoder_template(self, x, lstm_h, ):
        encoder_h = self.resolution_preserving_cnn_encoders(x)
        encoder_h, lstm_h = self.conv_lstm(encoder_h, lstm_h)
        return encoder_h, lstm_h

    def decoder_template(self, encoder_h, x_):
        output = self.pixel_cnn_decoders(encoder_h, x_)
        return output

    def build_model(self):
        lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state[0], self.initial_lstm_state[1])
        encoder_network_template = tf.make_template('vpn_encoder', self.encoder_template)
        decoder_network_template = tf.make_template('vpn_decoder', self.decoder_template)

        with tf.variable_scope('training_graph'):
            net_unwrap = []
            for i in range(self.config.truncated_steps):
                encoder_state, lstm_state = encoder_network_template(self.sequences[:, i], lstm_state, )
                step_out = decoder_network_template(encoder_state, self.sequences[:, i + 1])
                net_unwrap.append(step_out)

            self.final_lstm_state = lstm_state

        with tf.name_scope('wrap_out'):
            net_unwrap = tf.stack(net_unwrap)
            self.output = tf.transpose(net_unwrap, [1, 0, 2, 3, 4])

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.sequences[:, 1:]))

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.variable_scope('inference_graph'):
            encoder_state, lstm_state = encoder_network_template(self.inference_prev_frame, lstm_state)
            self.inference_lstm_state = lstm_state
            self.inference_outputinference_output = decoder_network_template(self.inference_encoder_state, self.inference_current_frame)
