#%%
import scipy.io as sio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from random_eraser import get_random_eraser

#%%
metadata = sio.loadmat('./metadata.mat')
allLandmarks = metadata['labelLandmarks'].astype(np.int8)
recordNum = metadata['labelRecNum'][0]
frameIndex = metadata['frameIndex'][0]

#%%
eraser = get_random_eraser(p=1, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                  v_l=0, v_h=255, pixel_level=True)


#%%
rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)

fig, axes = plt.subplots(nrows=2, ncols=4)
fig.subplots_adjust(hspace=0, wspace=0)
rowCount = 0
colCount = 0
for i in range(8):
    index = random.randint(0, 1320000)
    image = cv2.imread('/home/shared_for_databases/GazeCapture112/%05d/appleFace/%05d.jpg' % (recordNum[index], frameIndex[index]))[..., ::-1]
    # landmarks = allLandmarks[index]

    # draw = image.copy()
    # for landmark in landmarks:
    #     draw = cv2.circle(draw, (landmark[0], landmark[1]), 1, (125,255,125), -1)
    axes[rowCount][colCount].imshow(eraser(image))
    colCount += 1
    if colCount > 3:
        colCount = 0
        rowCount += 1 
plt.show()

#%%
# Draw landmarks
draw = image.copy()
# draw = cv2.polylines(draw, [np.int32(landmarks[:17].reshape(-1,1,2))], False, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[17:22].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[22:27].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[27:31].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[31:36].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[36:42].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[42:48].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[48:60].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[60:68].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[-16:-8].reshape(-1,1,2))], True, (125, 255, 125), 1)
# draw = cv2.polylines(draw, [np.int32(landmarks[-8:].reshape(-1,1,2))], True, (125, 255, 125), 1)
for landmark in landmarks:
    draw = cv2.circle(draw, (landmark[0], landmark[1]), 1, (125,255,125), -1)
plt.imshow(draw)



#%%
# View normalized images
# face_mean = sio.loadmat('./gaze-comp/mean_face_224.mat')['image_mean']
# image = cv2.resize(image, (224, 224))
# normalized_image = (image - face_mean).astype(int)
# plt.imshow(normalized_image)