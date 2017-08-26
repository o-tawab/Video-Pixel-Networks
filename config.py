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


class tiny_video_pixel_network_config:
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
    batch_size = 16
    truncated_steps = 10
    learning_rate = 3 * 1e-4
