class multiplicative_unit_config:
    mu_filters_num = 16


class residual_multiplicative_blocks_config:
    rmb_h1_filters_num = 16
    rmb_h4_filters_num = 32


class video_pixel_network_config:
    input_shape = [64, 64, 1]
    encoder_rmb_num = 8
    encoder_rmb_dilation = False
    encoder_rmb_dilation_scheme = [1, 2, 4, 8, 1, 2, 4, 8]
