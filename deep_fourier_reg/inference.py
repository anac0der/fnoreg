import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import utils
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from models import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse

vis = True
parser = argparse.ArgumentParser()
parser.add_argument('--fixed', type=str, help='Path to fixed image')
parser.add_argument('--moving', type=str, help='Path to moving image')
parser.add_argument('--output', type=str, help='Output path')
parser.add_argument('--weights', type=str, help='Path to model weights')

args = parser.parse_args()
weights_path = sys.argv[1]
# model = FourierNet(2, 2, 16, 256).cuda()
model = VxmModel(2, 2, 32).cuda()
model.load_state_dict(torch.load(args.weights))
model.eval()
transform = SpatialTransform()
fixed = torch.tensor(cv2.imread(args.fixed, cv2.IMREAD_GRAYSCALE) / 255)
moving = torch.tensor(cv2.imread(args.moving, cv2.IMREAD_GRAYSCALE) / 255)
fixed = fixed.unsqueeze(0).unsqueeze(0).cuda().float()
moving = moving.unsqueeze(0).unsqueeze(0).cuda().float()
grid, f_xy, X_Y = model(moving, fixed)
grid_x, grid_y = grid.squeeze(0)[:, :, 0].detach().cpu().numpy(), grid.squeeze(0)[:, :, 1].detach().cpu().numpy(), 
arr = [i for i in range(255) if i % 5 == 0]
grid_x = grid_x[arr, :]
grid_x = grid_x[:, arr]
grid_y = grid_y[arr, :]
grid_y = grid_y[:, arr]
fig, ax = plt.subplots()
utils.plot_grid(grid_x, grid_y, ax=ax, color="C0")
ax.invert_yaxis()
plt.savefig('field1.png')
loss = NCC(win=8)
print(loss(fixed, moving).item())
print(loss(fixed, X_Y).item())
print(MSE().forward(fixed, moving).item())
print(MSE().forward(fixed, X_Y).item())
fixed = fixed.squeeze(0).squeeze(0).cpu().numpy()
moving = moving.squeeze(0).squeeze(0).cpu().numpy()
moved = X_Y.detach().squeeze(0).squeeze(0).cpu().numpy()

cv2.imwrite(args.output, 255 * moved)
if vis:
    _, _, fixed_moving = utils.vis(fixed, moving)
    _, _, fixed_moved = utils.vis(fixed, moved)
    cv2.imwrite('fixed_moving.png', fixed_moving)
    cv2.imwrite('fixed_moved.png', fixed_moved)
    plot = plotter('inference', fixed, moving, moved, f_xy.squeeze(0).detach().cpu().numpy())
    cv2.imwrite('def.png', plot * 255)