# Author: Mark Gee
# Platform: keras
# Testing script for gaze tracker

from utils import ITrackerData
from utils.random_eraser import get_random_eraser
from models import ITrackerModel, ITrackerImprove, mobileIFT, SEITracker
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Testing the gaze tracker')
parser.add_argument('--model', help="Model to use (baseline, improved, seresnet, semobile, mobileift, semobileift)", required=True)
parser.add_argument('--weights', default=None, help="Path to weights to be loaded to start training from (optional).", required=True)
args = parser.parse_args()

# Define training parameters
batch_size = 20 # Change if out of cuda memory

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
# Change to test split as necessary
val_data_generator = ITrackerData.ITrackerData(batch_size, imSize=(224,224), split='val', mode=mode)

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

model.load_weights(args.weights)

# Evaluating
results = model.evaluate_generator(val_data_generator, verbose=1)
print('Results: \n: ', results)