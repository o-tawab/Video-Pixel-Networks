import numpy as np


class GenerateData:
    def __init__(self, config):
        self.config = config
        sequences = np.load(config.data_dir).transpose((1, 0, 2, 3))
        sequences = np.expand_dims(np.squeeze(sequences), 4)
        shuffled_idxs = np.random.shuffle(np.arange(sequences.shape[0]))
        sequences = sequences[shuffled_idxs]

        sequences = sequences[0]
        print(sequences.shape)

        self.train_sequences = sequences[:config.train_sequences_num]
        self.test_sequences = sequences[config.train_sequences_num:]

        # print(self.train_sequences.shape, self.test_sequences.shape)

    def next_batch(self):
        while True:
            idx = np.random.choice(self.config.train_sequences_num, self.config.batch_size)
            current_sequence = self.train_sequences[idx]

            yield current_sequence
            yield current_sequence

