import tensorflow as tf
from config import *
from model import VideoPixelNetworkModel
from data_generator import GenerateData
from trainer import Trainer
from logger import Logger

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('vpn_arch', "", """ full, mini or micro """)
tf.app.flags.DEFINE_boolean('train', True, """ train flag """)
tf.app.flags.DEFINE_boolean('overfitting', True, """ overfitting flag """)
tf.app.flags.DEFINE_boolean('load', True, """ model loading flag """)
tf.app.flags.DEFINE_integer('batch_size', 16, """ batch size for training """)
tf.app.flags.DEFINE_string('data_dir', "/tmp/vpn/mnist_test_seq.npy", """ data directory """)
tf.app.flags.DEFINE_string('exp_dir', "/tmp/vpn/", """ experiment directory """)


def main(_):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    if FLAGS.vpn_arch == 'full':
        config = video_pixel_network_config()
    elif FLAGS.vpn_arch == 'mini':
        config = mini_video_pixel_network_config()
    elif FLAGS.vpn_arch == 'micro':
        config = micro_video_pixel_network_config()
    else:
        Logger.info('wrong vpn_arch flag')
        return

    if FLAGS.train is not None:
        config.train = FLAGS.train
    if FLAGS.load is not None:
        config.load = FLAGS.load
    if FLAGS.batch_size is not None:
        config.batch_size = FLAGS.batch_size
    if FLAGS.overfitting is not None:
        config.overfitting = FLAGS.overfitting
        if FLAGS.overfitting:
            config.batch_size = 1
            config.train_sequences_num = 1
            config.iters_per_epoch = 1
    if FLAGS.data_dir is not None:
        config.data_dir = FLAGS.data_dir
    if FLAGS.exp_dir is not None:
        config.summary_dir = FLAGS.exp_dir + FLAGS.vpn_arch + '/'
        config.checkpoint_dir = config.summary_dir + 'checkpoints/'

    Logger.info('Starting building the model...')
    vpn = VideoPixelNetworkModel(config)
    data_generator = GenerateData(config)
    trainer = Trainer(sess, vpn, data_generator, config)
    Logger.info('Finished building the model')

    if config.train:
        trainer.train()
    trainer.test()


if __name__ == '__main__':
    tf.app.run()
