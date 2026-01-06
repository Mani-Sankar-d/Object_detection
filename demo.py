import torch
from PIL import Image
import numpy as np
import cv2

img = Image.open("D:/ml_data/VOCDevkit/cat.jpg").convert("RGB").resize((224,224))
# img = img.transpose(Image.FLIP_LEFT_RIGHT)
img_arr = np.array(img)
img_arr = cv2.resize(img_arr,(224,224))
img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
cv2.imshow("w",img_arr)

cv2.waitKey(0)