#%%
import os
try:
	os.chdir('/home_nfs/markgee/gaze-comp')
	print(os.getcwd())
except:
	pass

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#%%
import ITrackerData_3 as ITrackerData
import ITrackerImprove
import mobileIFT
from keras.preprocessing.image import ImageDataGenerator
from random_eraser import get_random_eraser

from keras.optimizers import SGD

import numpy as np
# from keras.utils import multi_gpu_model

#%%
epochs = 50
batch_size = 20 # Change if out of cuda memory
base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
lr = base_lr

#%%
model = mobileIFT.MobileIFTracker()
model.summary()
train_data_generator = ITrackerData.ITrackerData(batch_size, imSize=(224,224), mode='landmarks')
val_data_generator = ITrackerData.ITrackerData(batch_size, imSize=(224,224), split='val', mode='landmarks')

#%%
# image_gen = ImageDataGenerator(preprocessing_function=get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=1/0.5, r_2=0.5,
#                   v_l=0, v_h=255, pixel_level=True),
#                                data_format='channels_last',
#                                brightness_range=[0.5, 1.5])
# # image_gen_eye = ImageDataGenerator(preprocessing_function=get_random_eraser(p=0.1, s_l=0.01, s_h=0.01, r_1=0.3, r_2=1/0.3,
# #                   v_l=50, v_h=255, pixel_level=True),
# #                                zoom_range=[0.95, 1.05],
# #                                data_format='channels_last',
# #                                brightness_range=[0.5, 1.5])


# def create_aug_gen(in_gen):

#     for in_x, in_y in in_gen:
#         face = in_x['face']
#         gaze = in_y['gaze']
#         g_x = image_gen.flow(face, gaze,
#                              batch_size=face.shape[0])
#         x, _ = next(g_x)


#         yield {'face': x, 'grid': in_x['grid']}, {'gaze': gaze, 'lms': in_y['lms']}

# cur_gen = create_aug_gen(train_data_generator)

#%%
optimizer = SGD(lr=lr, momentum=momentum, decay=weight_decay)

losses = {
	"gaze": "mse",
	"lms": "mse",
}
lossWeights = {"gaze": 1.0, "lms": 1.0}

model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)

#%%
import scipy.io as sio
loss = dict()
val_loss = dict()
loss['lms_loss'] = []
loss['gaze_loss'] = []
val_loss['lms_loss'] = []
val_loss['gaze_loss'] = []

#loss = sio.loadmat('./mift_loss.mat')
#val_loss = sio.loadmat('./mift_val_loss.mat')


#%%
# loss = []
# val_loss = []
for epoch in range(epochs):
    history = model.fit_generator(generator=train_data_generator,
                            epochs=1,
                            steps_per_epoch=len(train_data_generator),
                            verbose=1,
                            validation_data=val_data_generator)
    loss['lms_loss'] = np.append(loss['lms_loss'], history.history['lms_loss'])
    loss['gaze_loss'] = np.append(loss['gaze_loss'], history.history['gaze_loss'])
    val_loss['lms_loss'] = np.append(val_loss['lms_loss'], history.history['val_lms_loss'])
    val_loss['gaze_loss'] = np.append(val_loss['gaze_loss'], history.history['val_gaze_loss'])
    sio.savemat('mift/mift_loss.mat', loss)
    sio.savemat('mift/mift_val_loss.mat', val_loss)
    model.save_weights('mift/mift_%d.h5' % (epoch+1))