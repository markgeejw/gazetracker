from keras.models import Model
from keras.layers import Dense, Activation, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D

from keras.layers import Input
import tensorflow as tf

def ITrackerImageModel():
    #features
    input = Input(shape=(224, 224, 3))
    x = input
    input = Conv2D(96, 11, strides=4)(input)
    input = Activation("relu")(input)
    input = MaxPooling2D(pool_size=3, strides=2)(input)

    input = BatchNormalization()(input)

    input = ZeroPadding2D(padding=2)(input)
    input = Conv2D(256, 5, strides=1)(input)
    input = Activation("relu")(input)
    input = MaxPooling2D(pool_size=3, strides=2)(input)

    input = BatchNormalization()(input)

    input = ZeroPadding2D(padding=1)(input)
    input = Conv2D(384, 3, strides=1)(input)
    input = Activation("relu")(input)
    input = Conv2D(64, 1, strides=1)(input)
    input = Activation("relu")(input)
    input = Flatten()(input)

    y = input    
    return Model(inputs=x, outputs=y)

def FaceImageModel(input):
    #Conv
    input = Conv2D(96, 11, strides=4)(input)
    input = Activation("relu")(input)
    input = MaxPooling2D(pool_size=3, strides=2)(input)

    input = BatchNormalization()(input)

    input = ZeroPadding2D(padding=2)(input)
    input = Conv2D(256, 5, strides=1)(input)
    input = Activation("relu")(input)
    input = MaxPooling2D(pool_size=3, strides=2)(input)

    input = BatchNormalization()(input)

    input = ZeroPadding2D(padding=1)(input)
    input = Conv2D(384, 3, strides=1)(input)
    input = Activation("relu")(input)
    input = Conv2D(64, 1, strides=1)(input)
    input = Activation("relu")(input)
    input = Flatten()(input)
    # FC
    input = Dense(128)(input)
    input = Activation("relu")(input)
    input = Dense(64)(input)
    input = Activation("relu")(input)

    return input

def FaceGridModel(input):
    #input = Flatten()(input)
    # FC
    input = Dense(256)(input)
    input = Activation("relu")(input)
    input = Dense(128)(input)
    input = Activation("relu")(input)

    return input

def ITrackerModel(image_shape=(224,224,3), grid_shape=25):
    eyes_left_input = Input(shape=image_shape, name='eyeL')
    eyes_right_input = Input(shape=image_shape, name='eyeR')
    face_input = Input(shape=image_shape, name='face')
    grid_input = Input(shape=(grid_shape*grid_shape,), name='grid')

    itracker_image_model = ITrackerImageModel()

    # eyes net
    x_eye_l = itracker_image_model(eyes_left_input)
    x_eye_r = itracker_image_model(eyes_right_input)
    x_eyes = concatenate([x_eye_l, x_eye_r])

    # eyes FC
    x_eyes = Dense(128)(x_eyes)
    x_eyes = Activation("relu")(x_eyes)

    # face net
    x_face = FaceImageModel(face_input)
    x_grid = FaceGridModel(grid_input)
    x_face_grid = concatenate([x_face, x_grid])

    x_face_grid = Dense(128)(x_face_grid)
    x_face_grid = Activation("relu")(x_face_grid)

    # cat all
    x = concatenate([x_eyes, x_face_grid])
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = Dense(2)(x)

    model = Model(inputs=[face_input, eyes_left_input, eyes_right_input, grid_input], outputs=x)

    return model



