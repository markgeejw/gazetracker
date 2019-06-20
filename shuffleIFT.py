# Author: Mark Gee
# Platform: keras
# Model for shuffleIFT
# Uses multi-task learning to learn facial landmarks and pupils
# and uses these landmarks to estimate eye gaze
# Based on shuffletnet v2

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, concatenate, ReLU, GlobalAveragePooling2D
from keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Add
from keras.layers import Input
from keras import backend as K
import numpy as np

from shuffle import block

def auxiliary_head_pose_net(input):
    x = input

    padded_x_1 = ZeroPadding2D(name='auxiliary_pad1')(x)
    x = Conv2D(128, 3, strides=2, name='auxiliary_conv1')(padded_x_1)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='auxiliary_bn1')(x)
    x = ReLU(name='auxiliary_relu1')(x)

    x = Conv2D(128, 3, name='auxiliary_conv2')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='auxiliary_bn2')(x)
    x = ReLU(name='auxiliary_relu2')(x)

    x = Add(name='auxiliary_add1')([padded_x_1, x])

    padded_x_2 = ZeroPadding2D(name='auxiliary_pad2')(x)
    x = Conv2D(32, 3, strides=2, name='auxiliary_conv3')(padded_x_2)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='auxiliary_bn3')(x)
    x = ReLU(name='auxiliary_relu3')(x)

    x = Conv2D(128, 7, name='auxiliary_conv4')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='auxiliary_bn4')(x)
    x = ReLU(name='auxiliary_relu4')(x)

    x = Add(name='auxiliary_add2')([padded_x_2, x])
    
    x = Dense(32, name='auxiliary_fc1')(x)
    x = ReLU(name='auxiliary_fc_relu')(x)

    x = Dense(3, name='head_pose')(x)

    return x


def ShuffleIFTracker(alpha=1, input_size=(112, 112, 3), grid_shape=25):
    grid_input = Input(shape=(grid_shape*grid_shape,), name='grid')
    face_input = Input(shape=input_size, name='face')
    out_dim_stage_two = {0.5:32, 1:64, 1.5:96, 2:128}
    out_channels_in_stage = np.array([1., 1., 2., 1.])
    out_channels_in_stage *= out_dim_stage_two[alpha]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage[-1] = 16  # last stage has always 16 output channels
    out_channels_in_stage = out_channels_in_stage.astype(int)
    print(out_channels_in_stage)

    # create shufflenet architecture
    x = Conv2D(out_channels_in_stage[0], 3, padding='same', use_bias=False, strides=1, name='conv1')(face_input)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='conv1_bn')(x)
    x = ReLU(6, name='conv1_relu')(x)

    # create stages containing shufflenet units beginning at stage 2
    x = block(x, out_channels_in_stage, repeat=3, bottleneck_ratio=alpha, stage=2)

    x = block(x, out_channels_in_stage, repeat=5, bottleneck_ratio=alpha, stage=3)

    x = block(x, out_channels_in_stage, repeat=3, bottleneck_ratio=alpha, stage=4)

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

    landmarks = Dense(168, name='lms')(s)

    x = concatenate([s, grid_input], name='grid_concat')

    x = Dense(128, name='fc1')(x)
    x = ReLU(6, name='fc1_relu')(x)

    gaze = Dense(2, name='gaze')(x)

    return Model(inputs=[face_input, grid_input], outputs=[landmarks, gaze], name='SIFT')

