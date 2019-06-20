'''
Data loader for the iTracker.
Modified from Petr Kellnhofer at: http://gazecapture.csail.mit.edu/
'''

import scipy.io as sio
from keras.utils import Sequence
from PIL import Image
import os
import os.path
import tensorflow as tf
import numpy as np
import cv2
import random

'''Define paths to required files'''
# DATASET_PATH = '/home/shared_for_databases/GazeCaptureProc'
DATASET_PATH = '/home/shared_for_databases/GazeCapture/dataset'
# DATASET_PATH = '/shared/datasets/GazeCaptureProc'
MEAN_PATH = './'
META_PATH = '../phone_metadata.mat'

# Loads metadata about dataset
def loadMetadata(filename, silent = False):
    try:
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


class ITrackerData(Sequence):
    def __init__(self, batch_size, split = 'train', mode = 'baseline', imSize=(224,224), gridSize=(25, 25)):
        """Generator for use with Keras models"""
        
        # Modes:
        # baseline  -   original ITracker architecture   (input: eyes, face, grid, output: gaze)
        # face      -   input just face                  (input: face, grid, output: gaze)
        # landmarks -   include landmarks in output      (input: face, grid, output: gaze, landmarks)
        self.mode = mode                # define modes for i/o
        self.n_channels = 3             # number of channels in images
        self.imSize = imSize            # size of images
        self.gridSize = gridSize        # size of facegrid, used to estimate position of face relative to camera
        self.batch_size = batch_size    # batch size for training

        # Load metadata and image means
        print('Loading iTracker dataset...')
        self.metadata = loadMetadata(META_PATH)
        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']

        # Define the training/val/test splits
        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']

        self.indices = np.argwhere(mask)[:,0]   # creates array of indices based on split mask
        random.shuffle(self.indices)            # shuffle the indices before training

        print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        '''Loads image'''
        im = cv2.imread(path)[...,::-1]
        if im is None:
            raise RuntimeError('Could not read image: ' + path)

        return im

    def transformImage(self, image, mean):
        '''Transform image and normalize by mean'''
        im = cv2.resize(image, self.imSize)
        im = im - mean[...,::-1]

        return im/255

    def makeGrid(self, params):
        '''Create face grid'''
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        '''Generator returns (input, output) for training'''
        # Create batches of face images and face grid
        imFaceBatch = np.empty((self.batch_size,) + self.imSize + (self.n_channels,))
        faceGridBatch = np.empty((self.batch_size, self.gridSize[0]*self.gridSize[1]))
        
        # Create batches of eye images if using baseline mode
        if self.mode == 'baseline':
            imEyeLBatch = np.empty((self.batch_size,) + self.imSize + (self.n_channels,))
            imEyeRBatch = np.empty((self.batch_size,) + self.imSize + (self.n_channels,))
        
        # Create batches of landmark coordinates for output if using landmarks mode
        if self.mode == 'landmarks':
            lmsBatch = np.empty((self.batch_size,168))
        
        # Create batches of gaze coordinates for output
        gazeBatch = np.empty((self.batch_size,2))

        # Fill batches
        for i in range(self.batch_size):
            # Obtain index for image, landmark and gaze retrieval
            idx = self.indices[index*self.batch_size+i]

            # Path to face image
            imFacePath = os.path.join(DATASET_PATH, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][idx], self.metadata['frameIndex'][idx]))
            
            # Transform image and save into batch
            imFace = self.loadImage(imFacePath)
            imFace = self.transformImage(imFace, self.faceMean)
            imFace = np.array(imFace)
            imFaceBatch[i] = imFace

            # Save gaze into batch
            gaze = np.array([self.metadata['labelDotXCam'][idx], self.metadata['labelDotYCam'][idx]], np.float32)
            gazeBatch[i] = gaze
            
            # Save grid into batch
            faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][idx,:])
            faceGridBatch[i] = faceGrid

            # Load landmarks if using landmarks mode
            if self.mode == 'landmarks':
                lms = np.array(self.metadata['labelLandmarks'][idx], np.int8)
                lmsBatch[i] = lms.flatten()

            # Load eyes if using baseline mode
            if self.mode == 'baseline':
                # Load eye images and transform
                imEyeLPath = os.path.join(DATASET_PATH, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][idx], self.metadata['frameIndex'][idx]))
                imEyeRPath = os.path.join(DATASET_PATH, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][idx], self.metadata['frameIndex'][idx]))
                imEyeL = self.loadImage(imEyeLPath)
                imEyeR = self.loadImage(imEyeRPath)
                imEyeL = self.transformImage(imEyeL, self.eyeLeftMean)
                imEyeR = self.transformImage(imEyeR, self.eyeRightMean)
                imEyeR = np.array(imEyeR)
                imEyeL = np.array(imEyeL)

                imEyeLBatch[i] = imEyeL
                imEyeRBatch[i] = imEyeR

        if self.mode == 'baseline':
            return {'face': imFaceBatch, 'eyeL': imEyeLBatch, 'eyeR': imEyeRBatch, 'grid': faceGridBatch}, gazeBatch
        elif self.mode == 'landmarks':
            return {'face': imFaceBatch, 'grid': faceGridBatch}, {'gaze': gazeBatch, 'lms': lmsBatch}
        elif self.mode == 'face':
            return {'face': imFaceBatch, 'grid': faceGridBatch}, gazeBatch

    def __len__(self):
        '''Length of generator'''
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        '''Updates indices after each epoch'''
        random.shuffle(self.indices)
