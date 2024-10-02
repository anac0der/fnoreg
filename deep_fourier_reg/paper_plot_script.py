import cv2
import numpy as np
import matplotlib.pyplot as plt


fixed = cv2.cvtColor(cv2.imread('fixed.png', cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
moving = cv2.cvtColor(cv2.imread('moving.png', cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)

sep = 255 * np.ones((int(fixed.shape[-3] * 0.01), fixed.shape[-3], 3))
image = np.vstack([np.rot90(fixed, -1), sep, np.rot90(moving, -1), sep, np.rot90(np.zeros_like(fixed))])
hsep = 255 * np.ones((image.shape[0], int(fixed.shape[-3] * 0.01), 3))

img_list = [
    'plot_fouriernet.png',
    'plot_deepunet.png',
    'plot_vxm-huge.png',
    'plot_TransMorph.png',
    'plot_fno.png',
    'plot_convfno.png'    
]

for img in img_list:
    img_arr = cv2.imread(img, cv2.IMREAD_COLOR)
    image = np.hstack([image, hsep, img_arr])

cv2.imwrite('plot_fields.png', image)