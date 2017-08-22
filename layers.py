import tensorflow as tf
import numpy as np


def masked_conv2d(inputs,
                  in_channels,  # num of channels in input
                  filters,
                  kernel_size,
                  mask_type,  # A, B or None
                  channels_masked,
                  padding='valid',
                  activation=None,
                  kernel_initializer=None,
                  name=None):

    with tf.variable_scope(name):
        weights_shape = (kernel_size, kernel_size, in_channels, filters)
        weights = tf.get_variable("weights", weights_shape,
                                  tf.float32, kernel_initializer)

        if mask_type is not None:
            center_h = kernel_size // 2
            center_w = kernel_size // 2

            mask = np.ones(weights_shape, dtype=np.float32)

            mask[center_h, center_w + 1:, :channels_masked, :] = 0.
            mask[center_h + 1:, :, :channels_masked, :] = 0.

            if mask_type == 'A':
                mask[center_h, center_w, :channels_masked, :] = 0.

            weights *= tf.constant(mask, dtype=tf.float32)

        outputs = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding=padding, name='outputs')

        biases = tf.get_variable("biases", [filters, ], tf.float32, tf.zeros_initializer())

        outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

        if activation:
            outputs = activation(outputs, name='outputs_with_fn')

        return outputs


class ActionLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, num_units, w_hv, w_hz, w_vh, w_va, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None):
        super(tf.nn.rnn_cell.BasicLSTMCell, self).__init__(_reuse=reuse)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

        self._w_hv = w_hv
        self._w_hz = w_hz
        self._w_vh = w_vh
        self._w_va = w_va

    def __call__(self, x, h, a, scope=None):
        with tf.variable_scope(self._scope or self.__class__.__name__):
            previous_memory, previous_output = h

            v = tf.matmul(self._w_vh, tf.transpose(previous_output, (1, 0))) * tf.matmul(self._w_va,
                                                                                         tf.transpose(a, (1, 0)))
            w_v = tf.matmul(self._w_hv, v)
            iv, fv, ov, cv = tf.split(w_v, 4, axis=0)
            w_z = tf.matmul(self._w_hz, tf.transpose(x))
            iz, fz, oz, cz = tf.split(w_z, 4, axis=0)

            i = tf.sigmoid(iv + iz)
            f = tf.sigmoid(fv + fz)
            o = tf.sigmoid(ov + oz)
            memory = f * tf.transpose(previous_memory, (1, 0)) + i * tf.tanh(cv + cz)
            output = o * tf.tanh(memory)

        return tf.transpose(output, (1, 0)), tf.contrib.rnn.LSTMStateTuple(tf.transpose(memory, (1, 0)),
                                                                           tf.transpose(output, (1, 0)))


def actionlstm_cell(scope, x, h, a, num_units, input_shape, action_dim,
                    initializer=tf.contrib.layers.xavier_initializer(), forget_bias=1.0,
                    activation=tf.tanh):
    with tf.name_scope(scope) as scope_:
        # Initialize the weights
        state_size = input_shape[1]

        w_hv = tf.get_variable('w_hv', [4 * num_units, 2 * num_units], initializer=initializer)
        w_hz = tf.get_variable('w_hz', [4 * num_units, state_size], initializer=initializer)
        w_vh = tf.get_variable('w_vh', [2 * num_units, num_units], initializer=initializer)
        w_va = tf.get_variable('w_va', [2 * num_units, action_dim], initializer=initializer)

        # init the cell
        cell = ActionLSTMCell(num_units, w_hv, w_hz, w_vh, w_va, forget_bias, activation)
        # call the cell
        if h is None:
            h = cell.zero_state(tf.shape(x)[0], tf.float32)

        output, state = cell(x, h, a)

    return output, state
