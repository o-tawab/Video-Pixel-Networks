import tensorflow as tf
from config import tiny_video_pixel_network_config
from model import VideoPixelNetworkModel
from data_generator import GenerateData
from trainer import Trainer
from logger import Logger
import os

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.DEBUG)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

config = tiny_video_pixel_network_config()
vpn = VideoPixelNetworkModel(config)
data_generator = GenerateData(config)
trainer = Trainer(sess, vpn, data_generator, config)

trainer.train()
trainer.test()