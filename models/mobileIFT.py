# Author: Mark Gee
# Platform: keras
# Model for mobileIFT
# Uses multi-task learning to learn facial landmarks and pupils
# and uses these landmarks to estimate eye gaze
# Credits to PFLD for inspiration for model

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, concatenate, ReLU, GlobalAveragePooling2D
from keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Add
from keras.layers import Input
from keras import backend as K

from .se import squeeze_excite_block

# Inverted residual block by mobilenetv2
def _inverted_res_block(inputs, filters, expansion, stride, alpha, block_id, use_se=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(inputs)[-1]
    pointwise_filters = int(filters * alpha)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    # Expand
    x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, name=prefix + 'expand')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
    x = ReLU(6, name=prefix + 'expand_relu')(x)

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(name=prefix + 'pad')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same' if stride == 1 else 'valid', use_bias=False, name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, name=prefix + 'project')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    # Use SE block if needed
    if use_se:
        x = squeeze_excite_block(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

# MobileIFTracker model
def MobileIFTracker(alpha=1, expansion=6, input_size=(224, 224, 3), grid_shape=25, use_se=False):
    face_input = Input(shape=input_size, name='face')
    grid_input = Input(shape=(grid_shape*grid_shape,), name='grid')

    # mobilenetv2 (PFLD implementation)
    x = ZeroPadding2D(name='conv1_pad')(face_input)
    x = Conv2D(64, 3, strides=2, name='conv1')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='conv1_bn')(x)
    x = ReLU(6, name='conv1_relu')(x)

    x = DepthwiseConv2D(3, depth_multiplier=alpha, name='dw_conv1')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='dwconv1_bn')(x)
    x = ReLU(6, name='dwconv1_relu')(x)

    # Group 1
    x = _inverted_res_block(x, filters=64, expansion=2, stride=2, alpha=alpha, block_id=1, use_se=use_se)
    x = _inverted_res_block(x, filters=64, expansion=2, stride=1, alpha=alpha, block_id=2, use_se=use_se)
    x = _inverted_res_block(x, filters=64, expansion=2, stride=1, alpha=alpha, block_id=3, use_se=use_se)
    x = _inverted_res_block(x, filters=64, expansion=2, stride=1, alpha=alpha, block_id=4, use_se=use_se)
    x = _inverted_res_block(x, filters=64, expansion=2, stride=1, alpha=alpha, block_id=5, use_se=use_se)

    # Group 2
    x = _inverted_res_block(x, filters=128, expansion=2, stride=2, alpha=alpha, block_id=6, use_se=use_se)

    # Group 3
    x = _inverted_res_block(x, filters=128, expansion=4, stride=1, alpha=alpha, block_id=7, use_se=use_se)
    x = _inverted_res_block(x, filters=128, expansion=4, stride=1, alpha=alpha, block_id=8, use_se=use_se)
    x = _inverted_res_block(x, filters=128, expansion=4, stride=1, alpha=alpha, block_id=9, use_se=use_se)
    x = _inverted_res_block(x, filters=128, expansion=4, stride=1, alpha=alpha, block_id=10, use_se=use_se)
    x = _inverted_res_block(x, filters=128, expansion=4, stride=1, alpha=alpha, block_id=11, use_se=use_se)
    x = _inverted_res_block(x, filters=128, expansion=4, stride=1, alpha=alpha, block_id=12, use_se=use_se)

    # Group 4
    x = _inverted_res_block(x, filters=16, expansion=2, stride=1, alpha=alpha, block_id=13, use_se=use_se)

    # features
    s1 = GlobalAveragePooling2D(name='s1_pool')(x)
    x = ZeroPadding2D(name='s1_pad')(x)
    x = Conv2D(32, 3, strides=2, name='s1_conv')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='s1_bn')(x)
    x = ReLU(6, name='s1_relu')(x)

    s2 = GlobalAveragePooling2D(name='s2_pool')(x)
    x = Conv2D(128, 7, name='s2_conv')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='s2_bn')(x)
    x = ReLU(6, name='s2_relu')(x)

    s3 = GlobalAveragePooling2D(name='s3_pool')(x)
    s = concatenate([s1, s2, s3], name='s_concat')

    # Landmarks network
    landmarks = Dense(256, name='lms_dense')(s)
    landmarks = ReLU(6, name='lms_relu')(landmarks)
    landmarks = Dense(168, name='lms')(landmarks)

    # Face grid network
    x_grid = Dense(256, name='grid_dense1')(grid_input)
    x_grid = ReLU(6, name='grid_relu1')(x_grid)
    x_grid = Dense(128, name='grid_dense2')(x_grid)
    x_grid = ReLU(6, name='grid_relu2')(x_grid)

    # Gaze network
    x = concatenate([s, x_grid], name='grid_concat')
    x = Dense(256, name='fc1')(x)
    x = ReLU(6, name='fc1_relu')(x)
    x = Dense(128, name='fc2')(x)
    x = ReLU(6, name='fc2_relu')(x)

    gaze = Dense(2, name='gaze')(x)

    return Model(inputs=[face_input, grid_input], outputs=[landmarks, gaze], name='MIFT')

