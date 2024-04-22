import os
import sys
from sklearn.model_selection import train_test_split
import utils
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from models import *
import metrics
import time
from tqdm import tqdm
import cv2

params = pd.read_json('params.json')
DATASET_SIZE = params['dataset_size'][0]
DOWNSAMPLE_SIZE = params['downsample_size'][0]
PATCH_SIZE = params['patch_size'][0]
DATASET_PATH = params['dataset_path'][0]
LANDMARKS_PATH =  params['landmarks_path'][0]
WEIGHTS_PATH = params['weights_path'][0]
DATASET_CSV_PATH = params['dataset_csv_path'][0]
    
is_landmarks = True

fixed_train, fixed_test, moving_train, moving_test, fixed_test_landmarks, moving_test_landmarks = utils.load_dataset(DATASET_SIZE, DOWNSAMPLE_SIZE, DATASET_PATH, DATASET_CSV_PATH, landmarks=is_landmarks, landmarks_path=LANDMARKS_PATH)

weights_path = sys.argv[1]
model = VxmModel(2, 2, 16)
model.load_state_dict(torch.load(weights_path))
model.eval()
test = utils.ArrayWrapper(fixed_test, moving_test, landmarks=is_landmarks, fixed_landmarks_src=fixed_test_landmarks, moving_landmarks_src=moving_test_landmarks)
test_gen = Data.DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=4)
initial_mrtre = []
mrtre_after_reg = []
robustness = []
reg_times = []
print('Computing metrics...')
for moving, fixed, moving_landmarks, fixed_landmarks in tqdm(test_gen):
    t = time.time()
    moving = moving.float()
    fixed = fixed.float()
    f_xy, X_Y = model(moving, fixed)
    reg_times.append(time.time() - t)
    fixed = fixed.numpy()
    fixed = np.squeeze(np.squeeze(fixed, axis=0), axis=0)
    moving = moving.numpy()
    moving = np.squeeze(np.squeeze(moving, axis=0), axis=0)
    fixed_landmarks = fixed_landmarks.squeeze(0).numpy()
    moving_landmarks = moving_landmarks.squeeze(0).numpy()
    # print(fixed_landmarks.shape, moving_landmarks.shape)
    f_xy = f_xy.squeeze(0).detach().numpy() * 512
    # print(f_xy.shape)
    moved_landmarks = utils.transform_landmarks(moving_landmarks, f_xy)
    # print(moved_landmarks.shape)
    # print(np.sum((moving_landmarks - moved_landmarks)**2, axis=None))   
    moved = X_Y.detach().numpy()
    moved = np.squeeze(np.squeeze(moved, axis=0), axis=0)
    image_diagonal = np.sqrt(fixed.shape[0] ** 2 + fixed.shape[1] ** 2)
    cv2.imwrite('fixed.png', 255 * fixed)
    cv2.imwrite('moving.png', 255 * moving)
    initial_mrtre.append(np.median(metrics.rTRE(moving_landmarks, fixed_landmarks, image_diagonal)))
    mrtre_after_reg.append(np.median(metrics.rTRE(moved_landmarks, fixed_landmarks, image_diagonal)))
    robustness.append(metrics.robustness(fixed_landmarks, moving_landmarks, moved_landmarks, image_diagonal)) 
metric1_name = 'MMrTRE'
metric2_name = "AMrTRE"
metric3_name = "robustness"
print(f'Metrics computed:')
print(f'Initial {metric1_name}: {np.median(initial_mrtre):.5f}')
print(f'Initial {metric2_name}: {np.mean(initial_mrtre):.5f}')
print('--- Fourier-Net ---')
print(f'Mean registration time: {np.mean(np.array(reg_times)):.5f} seconds')
print(f'{metric1_name} after registration: {np.median(mrtre_after_reg):.5f}')
print(f'{metric2_name} after registration: {np.mean(mrtre_after_reg):.5f}')
print(f'Average {metric3_name}: {np.mean(robustness):.5f}')
print(f'Median {metric3_name}: {np.median(robustness):.5f}')