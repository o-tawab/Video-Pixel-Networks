import numpy as np
from logger import Logger


class GenerateData:
    def __init__(self, config):
        self.config = config
        np.random.seed(123)
        sequences = np.load(config.data_dir).transpose((1, 0, 2, 3))
        sequences = np.expand_dims(np.squeeze(sequences), 4)
        shuffled_idxs = np.arange(sequences.shape[0])
        np.random.shuffle(shuffled_idxs)
        sequences = sequences[shuffled_idxs]

        Logger.debug(('data shape', sequences.shape))

        self.train_sequences = sequences[:config.train_sequences_num]
        self.test_sequences = sequences[config.train_sequences_num:]

    def next_batch(self):
        while True:
            idx = np.random.choice(self.config.train_sequences_num, self.config.batch_size)
            current_sequence = self.train_sequences[idx]

            return current_sequence[:, :self.config.truncated_steps + 1], current_sequence[:, self.config.truncated_steps:2 * self.config.truncated_steps + 1]

    def test_batch(self):
        while True:
            idx = np.random.choice(self.test_sequences.shape[0], self.config.batch_size)
            current_sequence = self.test_sequences[idx]

            return current_sequence[:, :self.config.truncated_steps + 1], current_sequence[:, self.config.truncated_steps:2 * self.config.truncated_steps + 1]
