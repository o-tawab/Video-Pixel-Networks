import tensorflow as tf
from config import tiny_video_pixel_network_config
from model import VideoPixelNetworkModel

tf.reset_default_graph()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

config = tiny_video_pixel_network_config()
vpn = VideoPixelNetworkModel(config)

print('fuck yeah')
