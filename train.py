# Author: Mark Gee
# Platform: keras
# Training script for gaze tracker

from utils import ITrackerData
from utils.random_eraser import get_random_eraser
from models import ITrackerModel, ITrackerImprove, mobileIFT, SEITracker
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Training the gaze tracker')
parser.add_argument('--model', help="Model to use (baseline, improved, seresnet, semobile, mobileift, semobileift)", required=True)
parser.add_argument('--epochs', default=1, help="Number of epochs to train (default: 1)")
parser.add_argument('--aug', default=None, help="Augmentations to use: none (default), brightness, erasing, both")
parser.add_argument('--weights', default=None, help="Path to weights to be loaded to start training from (optional).")
args = parser.parse_args()

# Define training parameters
batch_size = 20 # Change if out of cuda memory
lr = 0.0001

# Define the model
if args.model == 'baseline':
    model = ITrackerModel.ITrackerModel()
    mode = 'baseline'
elif args.model == 'improve':
    model = ITrackerImprove.ITrackerModel()
    mode = 'baseline'
elif args.model == 'seresnet':
    model = SEITracker.SEITracker()
    mode = 'face'
elif args.model == 'semobile':
    model = SEITracker.SEITracker(type='mobile')
    mode = 'face'
elif args.model == 'mobileift':
    model = mobileIFT.MobileIFTracker()
    mode = 'landmarks'
elif args.model == 'semobileift':
    model = mobileIFT.MobileIFTracker(use_se=True)
    mode = 'landmarks'

model.summary()

# Load data generators from ITrackerData
train_data_generator = ITrackerData.ITrackerData(batch_size, imSize=(224,224), mode=mode)
val_data_generator = ITrackerData.ITrackerData(batch_size, imSize=(224,224), split='val', mode=mode)

# Data augmentation
# Can only be used with SEITracker type models due to the input and output types
if args.aug is not None:
    if args.aug == 'brightness':
        image_gen = ImageDataGenerator(data_format='channels_last',
                                    brightness_range=[0.5, 1.5])
    elif args.aug == 'erase':
        image_gen = ImageDataGenerator(preprocessing_function=get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=1, r_2=1,
                        v_l=0, v_h=255, pixel_level=True),
                                    data_format='channels_last')
    elif args.aug == 'both':
        image_gen = ImageDataGenerator(preprocessing_function=get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=1, r_2=1,
                        v_l=0, v_h=255, pixel_level=True),
                                    data_format='channels_last',
                                    brightness_range=[0.5, 1.5])
    # Create augmented data generator using the above image_gen 
    def create_aug_gen(in_gen):
        for in_x, in_y in in_gen:
            face = in_x['face']
            gaze = in_y
            g_x = image_gen.flow(face, gaze,
                                batch_size=face.shape[0])
            x, _ = next(g_x)
            yield {'face': x, 'grid': in_x['grid']}, gaze

    aug_gen = create_aug_gen(train_data_generator)

# Define the optimizer
optimizer = Adam(lr=lr)

# Compile with loss weights if using MobileIFTracker since multiple outputs
if args.model == 'mobileift' or args.model == 'semobileift':
    losses = {
        "gaze": "mse",
        "lms": "mse",
    }
    lossWeights = {"gaze": 1.0, "lms": 1.0}
    model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights)
else:
    model.compile(optimizer=optimizer, loss='mse')

if args.weights is not None:
    model.load_weights(args.weights)

# Method to keep track of losses
# Commented out
# Use your favourite method (i.e. Callbacks, etc)
loss = []
val_loss = []

# Training
for epoch in range(args.epochs):
    history = model.fit_generator(generator=train_data_generator if args.aug is None else aug_gen,
                            epochs=1,
                            steps_per_epoch=len(train_data_generator),
                            verbose=1,
                            validation_data=val_data_generator)
    # Record losses
    # loss = np.append(loss, history.history['loss'])
    # val_loss = np.append(val_loss, history.history['val_loss'])
    # Save the file
    # np.save('improve2/improve_loss.npy', loss)
    # np.save('improve2/improve_val_loss.npy', val_loss)
    # Save the weights progressively
    # model.save_weights('improve2/improve_%d.h5' % (epoch+1))

# Save weights
model.save_weights('./model/output/gazetracker.h5')