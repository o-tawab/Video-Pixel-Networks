import tensorflow as tf
from config import tiny_video_pixel_network_config
from model import VideoPixelNetworkModel
from data_generator import GenerateData

tf.reset_default_graph()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

config = tiny_video_pixel_network_config()
# vpn = VideoPixelNetworkModel(config)

# summary_writer = tf.summary.FileWriter('/tmp/vpn', sess.graph)

data = GenerateData(config)
print('fuck yeah')
