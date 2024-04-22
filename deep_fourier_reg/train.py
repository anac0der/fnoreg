import os
from sklearn.model_selection import train_test_split
import utils
import dataloaders
import pandas as pd
import torch
import torch.utils.data as Data
from torchsummary import summary
from models import *
from tqdm import tqdm
from fourier_models import FFCAE, FFCUnet
from fno import FNOReg, MyFNO
import matplotlib.pyplot as plt
import argparse
import torchvision.transforms as transforms

params = pd.read_json('params.json')
torch.manual_seed(2002)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int,  default=0, help='GPU number')
parser.add_argument('--size', type=int, default=256, help='Size of smaller dim of data sample, optional')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
device = torch.device("cuda")
print(torch.cuda.current_device())

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

fixed_train, fixed_test, moving_train, moving_test, fixed_test_landmarks, moving_test_landmarks = dataloaders.load_dataset(DATASET_SIZE, DOWNSAMPLE_SIZE, DATASET_PATH, DATASET_CSV_PATH, landmarks=is_landmarks, landmarks_path=LANDMARKS_PATH)

fixed_train, fixed_val, moving_train, moving_val = train_test_split(fixed_train, moving_train, test_size=0.1, random_state=12)

train_config = None
exp_meta_desc = None
model_name = params["model_name"][0].lower()
model_cfg = params['model_config'][0][model_name]
train_config = params['train_config'][0]

if model_name == 'ffcunet':
    model = FFCUnet(model_cfg).cuda()
elif model_name == 'fno':
    model = MyFNO(model_cfg).cuda()
elif model_name == 'fnoreg':
    model = FNOReg(model_cfg).cuda()
elif model_name == 'ffcae':
    model = FFCAE(**model_cfg).cuda()
elif model_name == 'fouriernet':
    model = FourierNet(**model_cfg).cuda()
elif model_name == 'symnet':
    model = SYMNetFull(**model_cfg).cuda()
elif model_name == 'deepunet':
    model = DeepUNet2d(model_cfg).cuda()
else:
    raise Exception('Incorrect model name!')
# print(summary(model, [(1, PATCH_SIZE, PATCH_SIZE), (1, PATCH_SIZE, PATCH_SIZE)]))

loss_similarity = NCC(win=8)
# loss_similarity = MSE()
loss_smooth = smoothloss

transform = SpatialTransform()
for param in transform.parameters():
    param.requires_grad = False
    param.volatile = True

optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
# optimizer = torch.optim.SGD(model.parameters(), lr=train_config['lr'], momentum=0.99)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1)

seed = 2002
bs = train_config['batch_size']
train = dataloaders.Datagen(bs * 2000, fixed_train, moving_train, params, seed=seed)
val = dataloaders.Datagen(bs * 200, fixed_val, moving_val, params, seed=seed)
train_gen = Data.DataLoader(dataset=train, batch_size=bs, shuffle=True, num_workers=8)
val_gen = Data.DataLoader(dataset=val, batch_size=bs, shuffle=False, num_workers=4)
reg_param = train_config['reg_param']
save = True
train_losses = []
val_losses = []
resize_transform = transforms.Resize(size=(args.size, args.size))
for epoch in range(train_config['epochs']):
    loss_train = 0
    sim_loss = 0
    model.train()
    # train_gen = utils.Datagen(train_config['steps_per_epoch'], fixed_train, moving_train, params, seed=seed, batch_size=bs)
    # val_gen = utils.Datagen(train_config['validation_steps'], fixed_val, moving_val, params, seed=seed, batch_size=bs)
    print(f'Epoch {epoch} started...')
    train_steps = 0
    for moving, fixed in tqdm(train_gen):
        for i in range(2):
            moving = resize_transform(moving)
            fixed = resize_transform(fixed)
            moving = moving.cuda().float()
            fixed = fixed.cuda().float()
            # print(step)   
            # _, f_xy, X_Y = model(moving, fixed)

            f_xy = model(moving, fixed)
            _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 1))
            loss1 = loss_similarity(fixed, X_Y)
            loss5, _, _= loss_smooth(f_xy)
            
            loss = loss1 + reg_param * loss5
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            sim_loss += loss1.item()
            train_steps += 1
            moving, fixed = fixed, moving

    with torch.no_grad():
        loss_val = 0
        model.eval()
        val_steps = 0
        for moving, fixed in val_gen:
            for i in range(2):
                moving = resize_transform(moving)
                fixed = resize_transform(fixed)
                moving = moving.cuda().float()
                fixed = fixed.cuda().float()

                # _, f_xy, X_Y = model(moving, fixed)

                f_xy = model(moving, fixed)
                _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 1))

                loss1 = loss_similarity(fixed, X_Y)
                loss5, _, _ = loss_smooth(f_xy)

                loss = loss1 + reg_param * loss5
                loss_val += loss.item()
                val_steps += 1
                moving, fixed = fixed, moving

        # if epoch >= 20:
        scheduler.step()
        loss_val /= val_steps
        loss_train /= train_steps
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        print(f'{model_name, args.size}: Epoch {epoch}, train loss: {(loss_train):5f}, val loss: {(loss_val):5f}, learning rate: {scheduler.get_last_lr()[0]}')

        sim_loss /= train_steps

loss = 'ncc'
exp_name = f'{model_name}_hist_{args.size}'
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

print()
print(exp_name)