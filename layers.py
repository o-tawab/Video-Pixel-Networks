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


