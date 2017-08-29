import tensorflow as tf


class BaseTrainer:

    def __init__(self, sess, config, flags):
        self.sess = sess
        self.config = config
        self.flags = flags

        self.cur_epoch_tensor = None
        self.cur_epoch_input = None
        self.cur_epoch_assign_op = None
        self.global_step_tensor = None
        self.global_step_input = None
        self.global_step_assign_op = None

        self.summary_placeholders = {}
        self.summary_ops = {}
        self.scalar_summary_tags = self.config.scalar_summary_tags

        # init the global step , the current epoch and the summaries
        self.init_global_step()
        self.init_cur_epoch()
        self.init_summaries()

        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.summary_writer = tf.summary.FileWriter(self.flags.summary_dir, self.sess.graph)

        if self.flags.cont_train or (not self.flags.is_train):
            self.load()

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

    def init_summaries(self):
        with tf.variable_scope('train-summary'):
            for tag in self.scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

    def add_summary(self, step, summaries_dict=None):
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

    def save(self, verbose=0):
        if verbose == 1:
            print("saving..")
        self.saver.save(self.sess, self.flags.checkpoint_dir, self.global_step_tensor)
        if verbose == 1:
            print("Saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.flags.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded from the lastest checkpoint")

    def train(self):
        raise NotImplementedError("train function is not implemented in the trainer")

    def test(self):
        raise NotImplementedError("Test function is not implemented in the trainer")

    def train_n_test(self):
        raise NotImplementedError("Train and Test function is not implemented in the trainer")
