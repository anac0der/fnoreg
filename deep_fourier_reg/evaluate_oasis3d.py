import utils
import dataloaders
import pandas as pd
import torch
import torch.utils.data as Data
import numpy as np
import os
from models import *
from models_3d import *
from tqdm import tqdm
import time
from fno import MyFNO, FNOReg3d
import argparse
import json
from plot_utils import plotter, dft_amplitude
import cv2
import matplotlib.pyplot as plt
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int,  default=0, help='GPU number')
parser.add_argument('--config_file', type=str,  default='params_3d.json', help='JSON config file name')
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

test = dataloaders.ValidationOasis3d(list(range(394, 414)), OASIS_PATH, oasis_folders)

exp_folder_name = os.path.join(WEIGHTS_PATH, F'oasis_v_exp{args.exp_num}')
with open(os.path.join(exp_folder_name, 'metadata.json'), "r") as f:
    exp_metadata_str = f.read()
exp_metadata = json.loads(exp_metadata_str)
model_cfg = exp_metadata['model_config']
model_name = exp_metadata['model_name']
if model_name == 'fno':
    model = MyFNO(model_cfg).to(device)
elif model_name == 'fouriernet':
    model = FourierNet3d(**model_cfg).to(device)
    model.patch_size = [160, 192, 224]
elif model_name == 'fnoreg':
    model = FNOReg3d(model_cfg).to(device)
elif model_name == 'deepunet':
    model = DeepUNet3d(model_cfg).to(device)
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
transform = SpatialTransform3d()
for param in transform.parameters():
    param.requires_grad = False
    param.volatile = True

test_gen = Data.DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=2)
val_steps = 0
dice_val = []
initial_dice_val = 0
test_steps = 0
inference_times = []
print('Computing metrics...')
for moving, fixed, moving_labels, fixed_labels in tqdm(test_gen, ncols=100):
    moving = moving.to(device).float()
    fixed = fixed.to(device).float()

    t = time.time()
    # f_xy, X_Y = model(moving, fixed)
    f_xy = model(moving, fixed)
    _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 4, 1))
    inference_times.append(time.time() - t)
    _, warped_labels = transform(moving_labels.to(device).float(), f_xy.permute(0, 2, 3, 4, 1), mod='nearest')   
    for i in range(warped_labels.shape[0]):
        wl = warped_labels[i].detach().long().cpu().numpy().copy()
        fl = fixed_labels[i].detach().long().cpu().numpy().copy()
        ml = moving_labels[i].detach().long().cpu().numpy().copy()
        dice = utils.dice(wl, fl)
        initial_dice = utils.dice(ml, fl)
        dice_val.append(dice)
        initial_dice_val += initial_dice
    test_steps += warped_labels.shape[0]

mean_inference_time = sum(inference_times) / len(inference_times)
dice_val = np.array(dice_val)
print(f"--- Evaluation results for {model_name} ---")
print()
print(f'Mean initial dice: {(initial_dice_val / test_steps):.3f}')
print(f'Mean dice after registration: {(np.mean(dice_val)):.3f}')
print(f'Standard deviation of dice values: {(np.std(dice_val)):.3f}')
print(f'Mean inference time: {(mean_inference_time):.3f} seconds')