import dataloaders
import pandas as pd
import torch
import torch.utils.data as Data
import numpy as np
import os
from tqdm import tqdm
import time
import cv2
from plot_utils import * 
import argparse
import json
from utils import dice, count_parameters
import RaFD.TransMorph2D.utils as utils
from Baseline_registration_models.VoxelMorph.models import VxmDense_1, VxmDense_2, VxmDense_huge, SpatialTransformer
import Baseline_registration_models.VoxelMorph.utils as utils
import utils as ut

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int,  default=0, help='GPU number')
parser.add_argument('--config_file', type=str,  default='params_vxm.json', help='JSON config file name')
parser.add_argument('--exp_num', type=int,  default=0, help='Experiment number')
parser.add_argument('--ckpt_epoch', type=int,  default=-1, help='Epoch of checkpoint')
args = parser.parse_args()

params = pd.read_json(args.config_file)

OASIS_FOLDERS_PATH = params['oasis_folders_path'][0]
OASIS_PATH = params['oasis_path'][0]
WEIGHTS_PATH = params['exp_path'][0]

use_cuda = True
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
device = torch.device("cuda" if use_cuda else "cpu")

oasis_folders = []
with open(OASIS_FOLDERS_PATH, 'r') as f:
    for line in f:
        oasis_folders.append(line.strip('\n'))

print('Loading OASIS dataset...')
oasis_data, oasis_labels = dataloaders.load_oasis(OASIS_PATH, oasis_folders)
oasis_test = oasis_data[213:]
oasis_test_labels = oasis_labels[213:]

exp_folder_name = os.path.join(WEIGHTS_PATH, F'oasis_exp{args.exp_num}')
with open(os.path.join(exp_folder_name, 'metadata.json'), "r") as f:
    exp_metadata_str = f.read()
exp_metadata = json.loads(exp_metadata_str)
model_name = exp_metadata['model_name']

inshape = (160, 192)
if model_name == 'vxm-1':
    model = VxmDense_1(inshape=inshape).cuda()
elif model_name == 'vxm-2':
    model = VxmDense_2(inshape=inshape).cuda()
elif model_name == 'vxm-huge':
    model = VxmDense_huge(inshape=inshape).cuda()
else:
    raise Exception('Incorrect model name!')

count_parameters(model) 

if args.ckpt_epoch < 0:
    weights_path = os.path.join(exp_folder_name, 'weights.pth')
    model.load_state_dict(torch.load(weights_path))
else:
    weights_path = os.path.join(exp_folder_name, f'ckpt_epoch{args.ckpt_epoch}.pt')
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt['model_state_dict'])

model.transformer = SpatialTransformer(size=(160, 192)).cuda()
model.eval()

reg_model = utils.register_model((160, 192), 'nearest')
reg_model.to(device)

test = dataloaders.ValidationOasis(oasis_test, oasis_test_labels)
test_gen = Data.DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=2)
val_steps = 0
dice_val = []
initial_dice_val = 0
test_steps = 0
inference_times = []
j_neg = []
sdlogJ = []
print('Computing metrics...')
for moving, fixed, moving_labels, fixed_labels in tqdm(test_gen, ncols=100):
    moving = moving.to(device).float()
    fixed = fixed.to(device).float()
    x = torch.cat([moving, fixed], 1)

    t = time.time()
    X_Y, f_xy = model(x)
    inference_times.append(time.time() - t)
    f_xy_J = torch.clone(f_xy)
    J = ut.jacobian_determinant(f_xy_J.detach().cpu().numpy().squeeze(0))
    J_neg_mask = np.where(J < 0, np.ones_like(J), np.zeros_like(J))
    j_neg.append(100 * np.sum(J_neg_mask) / J.size)
    f = J.reshape((-1,))
    J_log = np.log(f[f > 0])
    sdlogJ.append(np.std(J_log))
    warped_labels = reg_model([moving_labels.to(device).float(), f_xy.float()])   
    for i in range(warped_labels.shape[0]):
        wl = warped_labels[i].detach().long().cpu().numpy().copy()
        fl = fixed_labels[i].detach().long().cpu().numpy().copy()
        ml = moving_labels[i].detach().long().cpu().numpy().copy()
        dice_v = dice(wl, fl)
        initial_dice = dice(ml, fl)
        dice_val.append(dice_v)
        initial_dice_val += initial_dice
    test_steps += warped_labels.shape[0]

#plotting last image
X_Y = X_Y.detach().cpu().numpy()
f_xy = f_xy.detach().cpu().numpy()

plot = plotter(X_Y, f_xy)
cv2.imwrite(f'plot_{model_name}.png', plot)

mean_inference_time = sum(inference_times) / len(inference_times)
dice_val = np.array(dice_val)
j_neg = np.array(j_neg)
print(f"--- Evaluation results for {model_name} ---")
print()
print(f'Mean initial dice: {(initial_dice_val / test_steps):.3f}')
print(f'Mean dice after registration: {(np.mean(dice_val)):.3f}')
print(f'Standard deviation of dice values: {(np.std(dice_val)):.3f}')
print(f'Mean percent of folded pixels: {(np.mean(j_neg)):.3f}')
print(f'Std of percent of folded pixels: {(np.std(j_neg)):.3f}')
print(f'Mean sdlogJ: {(np.mean(np.array(sdlogJ))):.3f}')
print(f'Std sdlogJ: {(np.std(np.array(sdlogJ))):.3f}')
print(f'Mean inference time: {(mean_inference_time):.3f} seconds')
