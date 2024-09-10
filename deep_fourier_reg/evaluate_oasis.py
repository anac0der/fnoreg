import utils
import dataloaders
import pandas as pd
import torch
import torch.utils.data as Data
import numpy as np
import os
from models import *
from tqdm import tqdm
import time
from fno import MyFNO, FNOReg
import argparse
import json
from plot_utils import plotter, dft_amplitude
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int,  default=0, help='GPU number')
parser.add_argument('--config_file', type=str,  default='params.json', help='JSON config file name')
parser.add_argument('--exp_num', type=int,  default=0, help='Experiment number')
parser.add_argument('--ckpt_epoch', type=int,  default=-1, help='Epoch of checkpoint')
args = parser.parse_args()

params = pd.read_json(args.config_file)

OASIS_FOLDERS_PATH = params['oasis_folders_path'][0]
OASIS_PATH = params['oasis_path'][0]
WEIGHTS_PATH = params['weights_path'][0]

use_cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
device = torch.device("cuda" if use_cuda else "cpu")

oasis_folders = []
with open(OASIS_FOLDERS_PATH, 'r') as f:
    for line in f:
        oasis_folders.append(line.strip('\n'))

oasis_test = dataloaders.ValidationOasis2d(range(213, 414), OASIS_PATH, oasis_folders)

exp_folder_name = os.path.join(WEIGHTS_PATH, F'oasis_exp{args.exp_num}')
with open(os.path.join(exp_folder_name, 'metadata.json'), "r") as f:
    exp_metadata_str = f.read()
exp_metadata = json.loads(exp_metadata_str)
model_cfg = exp_metadata['model_config']
model_name = exp_metadata['model_name']
if model_name == 'fno':
    model = MyFNO(model_cfg).to(device)
elif model_name == 'convfno':
    model = FNOReg(model_cfg).to(device)
elif model_name == 'fouriernet':
    model = FourierNet(**model_cfg).to(device)
    model.patch_size = (160, 192)
elif model_name == 'deepunet':
    model = DeepUNet2d(model_cfg).to(device)
else:
    raise Exception('Incorrect model name!')

utils.count_parameters(model)

if args.ckpt_epoch < 0:
    weights_path = os.path.join(exp_folder_name, 'weights.pth')
    model.load_state_dict(torch.load(weights_path))
else:
    weights_path = os.path.join(exp_folder_name, f'ckpt_epoch{args.ckpt_epoch}.pt')
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt['model_state_dict'])
    
model.eval()
transform = SpatialTransform()
for param in transform.parameters():
    param.requires_grad = False
    param.volatile = True

test_gen = Data.DataLoader(dataset=oasis_test, batch_size=1, shuffle=False, num_workers=2)
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

    t = time.time()
    # f_xy, X_Y = model(moving, fixed)
    f_xy = model(moving, fixed)
    f_xy_J = torch.clone(f_xy)
    f_xy_J[:, 0, :, :] *= 159 / 2
    f_xy_J[:, 1, :, :] *= 191 / 2
    J = utils.jacobian_determinant(f_xy_J.detach().cpu().numpy().squeeze(0))
    J_neg_mask = np.where(J < 0, np.ones_like(J), np.zeros_like(J))
    # print(J)
    j_neg.append(100 * np.sum(J_neg_mask) / J.size)
    f = J.reshape((-1,))
    J_log = np.log(f[f > 0])
    sdlogJ.append(np.std(J_log))
    _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 1))
    inference_times.append(time.time() - t)
    _, warped_labels = transform(moving_labels.to(device).float(), f_xy.permute(0, 2, 3, 1), mod='nearest')   
    for i in range(warped_labels.shape[0]):
        wl = warped_labels[i].detach().long().cpu().numpy().copy()
        fl = fixed_labels[i].detach().long().cpu().numpy().copy()
        ml = moving_labels[i].detach().long().cpu().numpy().copy()
        dice = utils.dice(wl, fl)
        initial_dice = utils.dice(ml, fl)
        dice_val.append(dice)
        initial_dice_val += initial_dice
    test_steps += warped_labels.shape[0]

#plotting last image
X_Y = torch.squeeze(X_Y, (0, 1)).detach().cpu().numpy()
fixed = torch.squeeze(fixed, (0, 1)).detach().cpu().numpy()
moving = torch.squeeze(moving, (0, 1)).detach().cpu().numpy()
f_xy = torch.squeeze(f_xy, 0).detach().cpu().numpy()

_, _, vis_before = utils.vis(fixed, moving)
_, _, vis_after = utils.vis(fixed, X_Y)
plot = plotter(X_Y, f_xy)
dft = dft_amplitude(f_xy)
cv2.imwrite(f'plot_{model_name}.png', plot)
cv2.imwrite(f'dft_{model_name}.png', dft)
cv2.imwrite('vis_before.png', vis_before)
cv2.imwrite('vis_after.png', vis_after)
cv2.imwrite('fixed.png', fixed * 255)
cv2.imwrite('moving.png', moving * 255)
cv2.imwrite(f'moved_{model_name}.png', X_Y * 255)

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