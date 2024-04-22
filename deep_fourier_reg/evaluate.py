import os
import sys
from sklearn.model_selection import train_test_split
import utils
import dataloaders
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from models import *
import metrics
import time
from tqdm import tqdm
from fno import FNOReg, MyFNO
import cv2
import matplotlib.pyplot as plt
from plot_utils import plotter, dft_amplitude

params = pd.read_json('params.json')
DATASET_SIZE = params['dataset_size'][0]
DOWNSAMPLE_SIZE = params['downsample_size'][0]
PATCH_SIZE = params['patch_size'][0]
DATASET_PATH = params['dataset_path'][0]
LANDMARKS_PATH =  params['landmarks_path'][0]
WEIGHTS_PATH = params['weights_path'][0]
DATASET_CSV_PATH = params['dataset_csv_path'][0]
    
weights_path = sys.argv[1]
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
model_name = 'deepunet'
model =FourierNet(**{
        "in_channel": 2,
        "start_channel": 32,
        "n_classes": 2,
        "patch_size": [
          256,
          256
        ]
      }).cuda()
model.load_state_dict(torch.load(weights_path))
model.eval()
utils.count_parameters(model)
# model = FourierNet(2, 2, 16, PATCH_SIZE)
# model = FFCAE(2, 2, 32, enc_blocks=3, latent_space_blocks=9, alpha = 0.5, fu_kernel=3).to(device)
# model = VxmModel(2, 2, 32, ffc=4).to(device)

interactive_mode = True
is_landmarks = True

fixed_train, fixed_test, moving_train, moving_test, fixed_test_landmarks, moving_test_landmarks = dataloaders.load_dataset(DATASET_SIZE, DOWNSAMPLE_SIZE, DATASET_PATH, DATASET_CSV_PATH, landmarks=is_landmarks, landmarks_path=LANDMARKS_PATH)


transform = SpatialTransform()
amount = 4000
test = dataloaders.Datagen(amount, fixed_test, moving_test, params, seed=2002, landmarks=is_landmarks, fixed_landmarks_src=fixed_test_landmarks, moving_landmarks_src=moving_test_landmarks)
test_gen = Data.DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=1)
initial_mrtre = []
mrtre_after_reg = []
initial_dice = []
dice_after_reg = []
robustness = []
reg_times = []
# elastix_mrtre_after_reg = []
# elastix_robustness = []
# elastix_reg_times = []
print('Computing metrics...')
# dice = Dice().loss
mse = MSE()
p = 0
mse_losses_before = []
mse_losses_after = []
for moving, fixed, moving_landmarks, fixed_landmarks in tqdm(test_gen):
    if interactive_mode:
        p = input()
    t = time.time()
    moving = moving.to(device).float()
    fixed = fixed.to(device).float()
    f_xy = model(moving, fixed)
    f_xy_inv = model(fixed, moving)
    reg_times.append(time.time() - t)
    _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 1))
    fixed_landmarks = fixed_landmarks.squeeze(0).cpu().numpy()
    # print(fixed_landmarks.shape)
    if fixed_landmarks.shape[0] < 5:
        continue
    mse_losses_before.append(mse(fixed, moving).item())
    mse_losses_after.append(mse(fixed, X_Y).item())
    fixed = fixed.cpu().numpy()
    fixed = np.squeeze(np.squeeze(fixed, axis=0), axis=0)
    moving_landmarks = moving_landmarks.squeeze(0).cpu().numpy()
    # f_xy_forward = f_xy.permute((0, 2, 3, 1)).detach().cpu().numpy().squeeze(0) * 127.5
    # f_xy_inv, _ = inverse_flow_max(f_xy_forward)
    f_xy = f_xy.detach().cpu().numpy() * ((PATCH_SIZE - 1) / 2)
    f_xy_inv = f_xy_inv.detach().cpu().numpy() * ((PATCH_SIZE - 1) / 2)
    # print(f_xy.max(), f_xy.min())
    # print(f_xy.shape)
    moved_landmarks, flow_values = utils.transform_landmarks(moving_landmarks, f_xy_inv.squeeze(0))  # print(moving_landmarks)
    # print(flow_values)
    moved = X_Y.detach().cpu().numpy()
    moved = np.squeeze(np.squeeze(moved, axis=0), axis=0)
    moving = moving.detach().cpu().numpy()
    moving = np.squeeze(np.squeeze(moving, axis=0), axis=0)
    image_diagonal = np.sqrt(fixed.shape[0] ** 2 + fixed.shape[1] ** 2)
    #t = time.time()
    #elastix_reg_result = elastix_non_rigid_registration(fixed, moving, fixed_landmarks)
    #elastix_reg_times.append(time.time() - t)
    #elastix_landmarks = elastix_parse_landmarks('outputpoints.txt', fixed_landmarks.shape[0])
    
    initial_mrtre.append(np.median(metrics.rTRE(moving_landmarks, fixed_landmarks, image_diagonal)))
    mrtre_after_reg.append(np.median(metrics.rTRE(moved_landmarks, fixed_landmarks, image_diagonal)))
    robustness.append(metrics.robustness(fixed_landmarks, moving_landmarks, moved_landmarks, image_diagonal)) 
    #elastix_mrtre_after_reg.append(np.median(metrics.rTRE(elastix_landmarks, fixed_landmarks, image_diagonal)))  
    #elastix_robustness.append(metrics.robustness(fixed_landmarks, moving_landmarks, elastix_landmarks, image_diagonal))
    if interactive_mode:
        utils.plot_landmarks(fixed, fixed_landmarks, 'fixed_wl.png')
        utils.plot_landmarks(moving, moving_landmarks, 'moving_wl.png', flow=True, flow_values=flow_values, fixed=True, fixed_landmarks=fixed_landmarks)
        utils.plot_landmarks(moved, moved_landmarks, 'moved_wl.png')
        _, _, fixed_moving = utils.vis(fixed, moving)
        _, _, fixed_moved = utils.vis(fixed, moved)
        cv2.imwrite('fixed_moving_ev.png', fixed_moving)
        cv2.imwrite('fixed_moved_ev.png', fixed_moved)

        # grid_x, grid_y = grid.squeeze(0)[:, :, 0].detach().cpu().numpy(), grid.squeeze(0)[:, :, 1].detach().cpu().numpy(), 
        # arr = [i for i in range(PATCH_SIZE) if i % 4 == 0]
        # grid_x = grid_x[arr, :]
        # grid_x = grid_x[:, arr]
        # grid_y = grid_y[arr, :]
        # grid_y = grid_y[:, arr]
        # fig, ax = plt.subplots()
        # utils.plot_grid(grid_x, grid_y, ax=ax, color="C0")
        # ax.invert_yaxis()
        # plt.savefig('field_ev.png')

        plot = utils.plotter('inference', fixed, moving, moved, f_xy.squeeze(0))
        cv2.imwrite('plot_ev.png', plot * 255)

        X_Y = torch.squeeze(X_Y, (0, 1)).detach().cpu().numpy()
        f_xy = f_xy.squeeze(0)

        _, _, vis_before = utils.vis(fixed, moving)
        _, _, vis_after = utils.vis(fixed, X_Y)
        plot = plotter(X_Y, f_xy / ((PATCH_SIZE - 1) / 2))
        dft = dft_amplitude(f_xy / ((PATCH_SIZE - 1) / 2))
        cv2.imwrite(f'plot_{model_name}.png', plot)
        cv2.imwrite(f'dft_{model_name}.png', dft)
        cv2.imwrite('vis_before.png', vis_before)
        cv2.imwrite('vis_after.png', vis_after)
        cv2.imwrite('fixed.png', fixed * 255)
        cv2.imwrite('moving.png', moving * 255)
        cv2.imwrite(f'moved_{model_name}.png', X_Y * 255)
        print(f'Initial MrTRE: {initial_mrtre[-1]}')
        print(f'MrTRE after registration: {mrtre_after_reg[-1]}')
        print(f'Robustness: {robustness[-1]}')

metric1_name = 'MMrTRE'
metric2_name = "AMrTRE"
metric3_name = "robustness"
# metric4_name = 'Dice'
# elastix_mrtre_after_reg.append(0)
# elastix_reg_times.append(0)
# elastix_robustness.append(0)
print(f'Metrics computed on {amount} patches:')
print(f'Initial {metric1_name}: {np.median(initial_mrtre):.5f}')
print(f'Initial {metric2_name}: {np.mean(initial_mrtre):.5f}')
print(f'Initial MSE: {np.mean(np.array(mse_losses_before)):.5f}')
# print(f'Initial mean {metric4_name}: {np.nanmean(initial_dice):.5f}')
print('--- Fourier-Net ---')
print(f'Mean registration time: {np.mean(np.array(reg_times)):.5f} seconds')
print(f'{metric1_name} after registration: {np.median(mrtre_after_reg):.5f}')
print(f'{metric2_name} after registration: {np.mean(mrtre_after_reg):.5f}')
print(f'Mean MSE after registration: {np.mean(np.array(mse_losses_after)):.5f}')
# print(f'Mean {metric4_name} after registration: {np.nanmean(dice_after_reg):.5f}')
print(f'Average {metric3_name}: {np.mean(robustness):.5f}')
print(f'Median {metric3_name}: {np.median(robustness):.5f}')
# print('--- Simple Elastix ---')
# print(f'Mean registration time: {np.mean(np.array(elastix_reg_times)):.5f} seconds')
# print(f'{metric1_name} after registration: {np.median(elastix_mrtre_after_reg):.5f}')
# print(f'{metric2_name} after registration: {np.mean(elastix_mrtre_after_reg):.5f}')
# print(f'Average {metric3_name}: {np.mean(elastix_robustness):.5f}')
# print(f'Median {metric3_name}: {np.median(elastix_robustness):.5f}')