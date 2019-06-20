#%%
import ITrackerData_3 as ITrackerData
import SEITracker
import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from random_eraser import get_random_eraser

#%%
epochs = 20
batch_size = 32 # Change if out of cuda memory

#%%
# model = ITrackerModel.ITrackerModel()
model = SEITracker.SEITracker(type='mobile')
model.summary()
train_data_generator = ITrackerData.ITrackerData(batch_size, mode='face')
val_data_generator = ITrackerData.ITrackerData(batch_size, split='val', mode='face')

#%%
image_gen = ImageDataGenerator(data_format='channels_last',
                               brightness_range=[0.5, 1.5])

def create_aug_gen(in_gen):

    for in_x, in_y in in_gen:
        face = in_x['face']
        gaze = in_y
        g_x = image_gen.flow(face, gaze,
                             batch_size=face.shape[0])
        x, _ = next(g_x)


        yield {'face': x, 'grid': in_x['grid']}, gaze

cur_gen = create_aug_gen(train_data_generator)

#%%
model.compile(optimizer='adam', loss='mean_squared_error')
# model.load_weights('./final/final_3.h5')
#%%
loss = []
val_loss = []
# loss = np.load('./final/final_loss.npy')
# val_loss = np.load('./final/final_val_loss.npy')
for epoch in range(epochs):
    history = model.fit_generator(generator=cur_gen,
                            epochs=1,
                            steps_per_epoch=len(train_data_generator),
                            verbose=1,
                            validation_data=val_data_generator)
    loss = np.append(loss, history.history['loss'])
    val_loss = np.append(val_loss, history.history['val_loss'])
    np.save('final/final_loss.npy', loss)
    np.save('final/final_val_loss.npy', val_loss)
    model.save_weights('final/final_%d.h5' % (epoch+1))


#%%
