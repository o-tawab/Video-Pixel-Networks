class video_pixel_network_config:
    input_shape = [64, 64, 1]

    # RMB config
    rmb_c = 128

    # Encoder config
    encoder_rmb_num = 8
    encoder_rmb_dilation = False
    encoder_rmb_dilation_scheme = [1, 2, 4, 8, 1, 2, 4, 8]

    # Decoder config
    decoder_rmb_num = 12

    # ConvLSTM config
    conv_lstm_filters = 256

    # Training config
    batch_size = 16
    truncated_steps = 10
    learning_rate = 3 * 1e-4


class mini_video_pixel_network_config:
    input_shape = [64, 64, 1]

    # RMB config
    rmb_c = 32

    # Encoder config
    encoder_rmb_num = 4
    encoder_rmb_dilation = False
    encoder_rmb_dilation_scheme = [1, 2, 4, 8]

    # Decoder config
    decoder_rmb_num = 6

    # ConvLSTM config
    conv_lstm_filters = 64

    # Training config
    overfitting = True
    batch_size = 1
    epochs_num = 50000
    iters_per_epoch = 1
    truncated_steps = 9
    learning_rate = 3 * 1e-4
    load = True

    # Data config
    data_dir = '/shared/Windows1/oabdelta/vpn/mnist_test_seq.npy'
    train_sequences_num = 1

    # tensorflow config
    max_to_keep = 3
    test_every = 100
    checkpoint_dir = '/shared/Windows1/oabdelta/vpn/mini/checkpoints/'
    summary_dir = '/shared/Windows1/oabdelta/vpn/mini/'


class micro_video_pixel_network_config:
    input_shape = [64, 64, 1]

    # RMB config
    rmb_c = 32

    # Encoder config
    encoder_rmb_num = 2
    encoder_rmb_dilation = False
    encoder_rmb_dilation_scheme = [1, 2, 4, 8]

    # Decoder config
    decoder_rmb_num = 3

    # ConvLSTM config
    conv_lstm_filters = 64

    # Training config
    overfitting = True
    batch_size = 1
    epochs_num = 50000
    iters_per_epoch = 1
    truncated_steps = 9
    learning_rate = 3 * 1e-4
    load = True

    # Data config
    data_dir = '/shared/Windows1/oabdelta/vpn/mnist_test_seq.npy'
    train_sequences_num = 1

    # tensorflow config
    max_to_keep = 3
    test_every = 100
    checkpoint_dir = '/shared/Windows1/oabdelta/vpn/micro/checkpoints/'
    summary_dir = '/shared/Windows1/oabdelta/vpn/micro/'
