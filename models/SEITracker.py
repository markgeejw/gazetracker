
# Author: Mark Gee
# Platform: keras
# Model for SENet versions of ITracker
# Uses SENets for feature extraction to then predict gaze coordinates

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, concatenate, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.layers import Input

from .senet_keras import se_mobilenets, se_resnet

# SENet version of ITracker
def SEITracker(type='resnet', image_shape=(224, 224, 3), grid_shape=25):
    face_input = Input(shape=image_shape, name='face')
    grid_input = Input(shape=(grid_shape*grid_shape,), name='grid')
    # SE-ResNet
    if type == 'resnet':
        senet = se_resnet.SEResNet18(input_shape=(224,224,3), include_top=False, pooling='avg')
    # SE-MobileNet
    elif type == 'mobile':
        senet = se_mobilenets.SEMobileNet(input_shape=(224,224,3), include_top=False, pooling='avg')

    # Face features
    features = senet(face_input)

    # Face grid features
    x_grid = Dense(256)(grid_input)
    x_grid = Activation("relu")(x_grid)
    x_grid = Dense(128)(x_grid)
    x_grid = Activation("relu")(x_grid)

    # Concat and FC
    x = concatenate([features, x_grid])
    x = Dense(128)(x)
    x = Activation('relu')(x)

    x = Dense(2)(x)

    model = Model(inputs=[face_input, grid_input], outputs=x)

    return model

