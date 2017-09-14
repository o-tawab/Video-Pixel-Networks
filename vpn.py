import tensorflow as tf
from config import mini_video_pixel_network_config
from model import VideoPixelNetworkModel
from data_generator import GenerateData
from trainer import Trainer
from logger import Logger


def main():
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    Logger.info('Starting building the model...')
    config = mini_video_pixel_network_config()
    vpn = VideoPixelNetworkModel(config)
    data_generator = GenerateData(config)
    trainer = Trainer(sess, vpn, data_generator, config)
    Logger.info('Finished building the model')

    if config.train:
        trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()
