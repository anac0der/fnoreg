import os
from sklearn.model_selection import train_test_split
import utils
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torchsummary import summary
from models import *
from ffc_models import FFCAE
import cv2
import matplotlib.pyplot as plt
import numpy as np

params = pd.read_json('params.json')
DATASET_SIZE = params['dataset_size'][0]
DOWNSAMPLE_SIZE = params['downsample_size'][0]
PATCH_SIZE = params['patch_size'][0]
DATASET_PATH = params['dataset_path'][0]
LANDMARKS_PATH =  params['landmarks_path'][0]
WEIGHTS_PATH = params['weights_path'][0]
DATASET_CSV_PATH = params['dataset_csv_path'][0]
train_config = params['train_config'][0]

use_cuda = True

device = torch.device("cuda" if use_cuda else "cpu")
print(torch.cuda.current_device())

is_landmarks = False

fixed_train, fixed_test, moving_train, moving_test, fixed_test_landmarks, moving_test_landmarks = utils.load_dataset(DATASET_SIZE, DOWNSAMPLE_SIZE, DATASET_PATH, DATASET_CSV_PATH, landmarks=is_landmarks, landmarks_path=LANDMARKS_PATH)
fixed_train, fixed_val, moving_train, moving_val = train_test_split(fixed_train, moving_train, test_size=0.1, random_state=12)

fixed_anhir = np.concatenate([fixed_train, fixed_test])
moving_anhir = np.concatenate([moving_train, moving_test])

start_channels = train_config['start_channels']
model = VxmModel(2, 2, start_channels, ffc=0).cuda()
print(summary(model, [(1, DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), (1, DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE)]))
# loss_similarity = NCC(win=32)
loss_similarity = MSE()
loss_smooth = smoothloss

transform = SpatialTransform()
for param in transform.parameters():
    param.requires_grad = False
    param.volatile = True

optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
# optimizer = torch.optim.SGD(model.parameters(), lr=train_config['lr'], momentum=0.99)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

seed = 2002
bs = train_config['batch_size']
train = utils.ArrayWrapper(fixed_anhir, moving_anhir)
val = utils.ArrayWrapper(fixed_val, moving_val)
train_gen = Data.DataLoader(dataset=train, batch_size=bs, shuffle=True, num_workers=4)
val_gen = Data.DataLoader(dataset=val, batch_size=bs, shuffle=False, num_workers=2)
reg_param = train_config['reg_param']
save = True
train_losses = []
val_losses = []
for epoch in range(train_config['epochs']):
    loss_train = 0
    model.train()
    # train_gen = utils.Datagen(train_config['steps_per_epoch'], fixed_train, moving_train, params, seed=seed, batch_size=bs)
    # val_gen = utils.Datagen(train_config['validation_steps'], fixed_val, moving_val, params, seed=seed, batch_size=bs)
    print(f'Epoch {epoch} started...')
    train_steps = 0 
    for moving, fixed in train_gen:
        for i in range(2):
            moving = moving.cuda().float()
            fixed = fixed.cuda().float()
            # print(moving.shape)
            # print(step)   
            _, f_xy, X_Y = model(moving, fixed)
            # f_xy = model(moving, fixed)
            # _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 1))
            loss1 = loss_similarity(fixed, X_Y)
            loss5 = loss_smooth(f_xy)
            
            loss = loss1 + reg_param * loss5
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            train_steps += 1
            moving, fixed = fixed, moving
    with torch.no_grad():
        loss_val = 0
        model.eval()
        val_steps = 0
        for moving, fixed in val_gen:
            for i in range(2):
                moving = moving.cuda().float()
                fixed = fixed.cuda().float()

                _, f_xy, X_Y = model(moving, fixed)
                # f_xy = model(moving, fixed)
                # _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 1))

                loss1 = loss_similarity(fixed, X_Y)
                loss5 = loss_smooth(f_xy)

                loss = loss1 + reg_param * loss5
                loss_val += loss.item()
                val_steps += 1
                moving, fixed = fixed, moving
        if epoch >= 20:
            scheduler.step()
        loss_val /= val_steps
        loss_train /= train_steps
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        print(f'Epoch {epoch}, train loss: {(loss_train):5f}, val loss: {(loss_val):5f}, learning rate: {scheduler.get_last_lr()[0]}')

loss = 'ncc'
exp_name = f'ANHIR_FFCAE_121023_size={DOWNSAMPLE_SIZE}_{loss}_lr={train_config["lr"]}_reg_param={train_config["reg_param"]}_C={train_config["start_channels"]}_batch={bs}_epochs={train_config["epochs"]}_bw=64'
weights_path = os.path.join(WEIGHTS_PATH, exp_name + '.pth')
torch.save(model.state_dict(), weights_path)

loss_plot_name = os.path.join(WEIGHTS_PATH, exp_name + '.png')
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label = 'val loss')
plt.xlabel('Epoch number')
plt.ylabel(f'{loss} loss')
plt.ylim((-1, 1))
plt.legend()

plt.savefig(loss_plot_name)