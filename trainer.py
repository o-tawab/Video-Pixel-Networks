import tensorflow as tf
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, sess, model, data, config):
        self.sess = sess
        self.model = model
        self.data = data
        self._config = config

        self.cur_epoch_tensor = None
        self.cur_epoch_input = None
        self.cur_epoch_assign_op = None
        self.global_step_tensor = None
        self.global_step_input = None
        self.global_step_assign_op = None

        self.summary_placeholders = {}
        self.summary_ops = {}
        self.scalar_summary_tags = self._config.scalar_summary_tags

        # init the global step , the current epoch and the summaries
        self.init_global_step()
        self.init_cur_epoch()
        self.init_summaries()

        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.saver = tf.train.Saver(max_to_keep=self._config.max_to_keep)
        self.summary_writer = tf.summary.FileWriter(self._config.summary_dir, self.sess.graph)

        if self._config.load:
            self.load()

    def save(self):
        self.saver.save(self.sess, self._config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self._config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    # summaries init
    def init_summaries(self):
        with tf.variable_scope('train-summary'):
            for tag in self.scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            for tag in self._config.image_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', [None, 400, 300, 3], name=tag)
                self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=16)

        with tf.variable_scope('test-summary'):
            for tag in self._config.test_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            for tag in self._config.test_image_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', [None, 400, 300, 3], name=tag)
                self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=16)

    def train(self):

        initial_lstm_state = np.zeros((2, self._config.batch_size, 512))

        print("Training Finished")

    def test(self, cur_it):
        initial_lstm_state = np.zeros((2, self.data.xtest.shape[0], 512))
