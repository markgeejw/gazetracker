#%%
import os
import numpy as np
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir('/home_nfs/markgee/gaze-comp')

# #%%
# # Report results
# val_losses = []
# baseline_val_loss = np.load('./baseline2/baseline_val_loss.npy')
# val_losses.append(baseline_val_loss)
# print('Baseline: \n', baseline_val_loss)
# improve_val_loss = np.load('./improve2/improve_val_loss.npy')
# val_losses.append(improve_val_loss)
# print('Improve: \n', improve_val_loss)

# mift_val_loss = sio.loadmat('./mift/mift_val_loss.mat')['gaze_loss'][0]
# val_losses.append(mift_val_loss)
# print('mift: \n', mift_val_loss)
# semift_val_loss = sio.loadmat('./semift/semift_val_loss.mat')['gaze_loss'][0]
# val_losses.append(semift_val_loss)
# print('SEmift: \n', semift_val_loss)
# seresnet_val_loss = np.load('./seresnet/seresnet_val_loss.npy')
# val_losses.append(seresnet_val_loss)
# print('SEResnet: \n', seresnet_val_loss)
# semobile_val_loss = np.load('./semobile/semobile_val_loss.npy')
# val_losses.append(semobile_val_loss)
# print('SEMobile: \n', semobile_val_loss)

# #%%
# val_losses = []
# final_val_loss = np.load('./final/final_val_loss.npy')
# val_losses.append(final_val_loss)
# print('Final: \n', final_val_loss)
# final2_val_loss = np.load('./final2/final_val_loss.npy')
# val_losses.append(final2_val_loss)
# print('Final2: \n', final2_val_loss)

# losses = []
# final_loss = np.load('./final/final_loss.npy')
# losses.append(final_val_loss)
# print('Final: \n', final_loss)
# final2_loss = np.load('./final2/final_loss.npy')
# losses.append(final2_loss)
# print('Final2: \n', final2_loss)


# #%%
# losses = []
# baseline_loss = np.load('./baseline2/baseline_loss.npy')
# losses.append(baseline_loss)
# print('Baseline: \n', baseline_loss)
# improve_loss = np.load('./improve2/improve_loss.npy')
# losses.append(improve_loss)
# print('Improve: \n', improve_loss)

# mift_loss = sio.loadmat('./mift/mift_loss.mat')['gaze_loss'][0]
# losses.append(mift_loss)
# print('mift: \n', mift_loss)
# semift_loss = sio.loadmat('./semift/semift_loss.mat')['gaze_loss'][0]
# losses.append(semift_loss)
# print('SEmift: \n', semift_loss)
# seresnet_loss = np.load('./seresnet/seresnet_loss.npy')
# losses.append(seresnet_loss)
# print('SEResnet: \n', seresnet_loss)
# semobile_loss = np.load('./semobile/semobile_loss.npy')
# losses.append(semobile_loss)
# print('SEMobile: \n', semobile_loss)

# #%%
# alphabets = ["a", "b", "c", "d", "e", "f"]
# models = ["Baseline", "Improved", "MobileIFTracker", "SE-MobileIFTracker", "SE-ResNet-ITracker", "SE-Mobile-ITracker"]
# #%%
# import matplotlib.pyplot as plt

# plt.style.use('classic')

# #%%
# fig, axes = plt.subplots(2, 3)
# row = 0
# col = 0
# fig.subplots_adjust(hspace=0.5)
# fig.set_figwidth(15)
# for model in range(len(losses)):
#     axes[model//3][model%3].plot(losses[model])
#     axes[model//3][model%3].plot(val_losses[model])
#     axes[model//3][model%3].legend(['Train', 'Val'])
#     axes[model//3][model%3].set_title('(%s) %s' % (alphabets[model], models[model]))
#     axes[model//3][model%3].set_ylabel('Loss')
#     axes[model//3][model%3].set_xlabel('Epoch')
#     axes[model//3][model%3].set_ylim(bottom=0)

#%%
import ITrackerModel
import ITrackerImprove
# import ITrackerData_3 as ITrackerData
import SEITracker
# import mobileIFT
# import semobileIFT
import time
#%%
model = SEITracker.SEITracker(type='mobile')
# model = ITrackerImprove.ITrackerModel()
model.summary()
# model.compile(loss='mse', optimizer='adam')
#%%
start = time.time()
model.predict({'face': np.empty((1, 224, 224, 3)), 'grid': np.empty((1,625))})
end = time.time()
print(end - start)
# model.load_weights('./semobile/semobile_14.h5')

# #%%
# model = mobileIFT.MobileIFTracker()

# #%%
# test_data = ITrackerData.ITrackerData(256, split='test')

# #%%
# model.compile(optimizer='adam', loss='mse')

# #%%
# result = model.evaluate_generator(test_data, verbose=1)
# print(result)
