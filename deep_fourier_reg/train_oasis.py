import os
import utils
import dataloaders
import pandas as pd
import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from models import *
from losses import * 
from fourier_models import FFCUnet,FFCAE
from fno import MyFNO, FNOReg
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import re
import torchvision.transforms as transforms

torch.manual_seed(2002)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int,  default=0, help='GPU number')
parser.add_argument('--config_file', type=str,  default='params.json', help='JSON config file name')
parser.add_argument('--exp_num', type=int,  default=-1, help='Exp number to retrain, optional')
parser.add_argument('--ckpt_epoch', type=int, default=-1, help='Checkpoint epoch number')
parser.add_argument('--size', type=int, default=160, help='Size of smaller dim of data sample, optional')
args = parser.parse_args()

params = pd.read_json(args.config_file)
WEIGHTS_PATH = params['weights_path'][0]
OASIS_FOLDERS_PATH = params['oasis_folders_path'][0]
OASIS_PATH = params['oasis_path'][0]

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
device = torch.device("cuda")
print(torch.cuda.current_device())

oasis_folders = []
with open(OASIS_FOLDERS_PATH, 'r') as f:
    for line in f:
        oasis_folders.append(line.strip('\n'))

train_config = None
exp_meta_desc = None
model_name = params["model_name"][0].lower()
if args.exp_num < 0:
    model_cfg = params['model_config'][0][model_name]
    train_config = params['train_config'][0]
else:
    exp_folder_name = os.path.join(WEIGHTS_PATH, F'oasis_exp{args.exp_num}')
    with open(os.path.join(exp_folder_name, 'metadata.json'), "r") as f:
        exp_metadata_str = f.read()
    exp_metadata = json.loads(exp_metadata_str)
    model_cfg = exp_metadata['model_config']
    train_config = exp_metadata['train_config']
    exp_meta_desc = exp_metadata['description']

if model_name == 'ffcunet':
    model = FFCUnet(model_cfg).cuda()
elif model_name == 'fno':
    model = MyFNO(model_cfg).cuda()
elif model_name == 'convfno':
    model = FNOReg(model_cfg).cuda()
elif model_name == 'fouriernet':
    model = FourierNet(**model_cfg).cuda()
elif model_name == 'deepunet':
    model = DeepUNet2d(model_cfg).cuda()
else:
    raise Exception('Incorrect model name!')

utils.count_parameters(model)

transform = SpatialTransform().cuda()
for param in transform.parameters():
    param.requires_grad = False
    param.volatile = True
loss_similarity = MSE()
loss_smooth = smoothloss

optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
if train_config['lr_scheduler']:
    name = train_config['lr_scheduler']
    sch_params = train_config['lr_scheduler_params'][name]
    if name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sch_params)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10 ** 8, gamma=1) # constant LR on all training time

seed = 2002
bs = train_config['batch_size']
train = dataloaders.TrainOasis2d(list(range(201)), OASIS_PATH, oasis_folders)
val = dataloaders.ValidationOasis2d(list(range(201, 213)), OASIS_PATH, oasis_folders)
train_gen = Data.DataLoader(dataset=train, batch_size=bs, shuffle=True, num_workers=4, pin_memory = True)
val_gen = Data.DataLoader(dataset=val, batch_size=bs, shuffle=False, num_workers=2, pin_memory = True)
reg_param = train_config['reg_param']
train_losses = []
val_losses = []
dice_progress = []

exp_config = params['exp_config'][0]
if args.exp_num < 0:
    exp_dirs = []
    for _, dirs, _ in os.walk(WEIGHTS_PATH):
        for dir in dirs:
            if re.search('oasis_exp.', dir):
                dir = re.sub('[^0-9]', '0', dir)
                exp_dirs.append(int(dir))
    n = sorted(exp_dirs)[-1] + 1
else:
    n = args.exp_num
exp_folder_name = os.path.join(WEIGHTS_PATH, f'oasis_exp{n}')
try:
    os.mkdir(exp_folder_name)
except Exception:
    print(f'Folder for experiment {n} is already exists!')

exp_description = exp_config['description'] if not exp_meta_desc else exp_meta_desc

print(f'Experiment {n}: {exp_description}')
exp_metadata = dict()
exp_metadata['description'] = exp_description
exp_metadata['start time'] = str(datetime.now())
exp_metadata['model_config'] = model_cfg
exp_metadata['model_name'] = model_name
exp_metadata['train_config'] = train_config
json_metadata = json.dumps(exp_metadata)
with open(os.path.join(exp_folder_name, 'metadata.json'), "w") as f:
    f.write(str(json_metadata))

init_epoch = 0
if args.exp_num >= 0 and args.ckpt_epoch >= 0:
    try:
        ckpt = torch.load(os.path.join(exp_folder_name, f'ckpt_epoch{args.ckpt_epoch}.pt'))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        init_epoch = ckpt['epoch'] + 1
    except Exception as e:
        raise Exception(str(e))

writer = SummaryWriter(log_dir=os.path.join(exp_folder_name, "./logs"))
size = (args.size, int(192 / 160 * args.size))
resize_transform = transforms.Resize(size=size)
resize_labels_transform = transforms.Resize(size=size, interpolation=transforms.InterpolationMode.NEAREST)
for epoch in range(init_epoch, train_config['epochs']):
    loss_train = 0
    model.train()
    print(f'Epoch {epoch} started...')
    train_steps = 0 
    for moving, fixed in tqdm(train_gen, ncols=100):
        moving = resize_transform(moving)
        fixed = resize_transform(fixed)
        moving = moving.cuda().float()
        fixed = fixed.cuda().float()  
        f_xy = model(moving, fixed)
        _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 1))
        loss1 = loss_similarity(fixed, X_Y)
        loss5, _, _ = loss_smooth(f_xy)
        
        loss = loss1 + reg_param * loss5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        train_steps += 1
        
    if epoch % 10 == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                }, os.path.join(exp_folder_name, f'ckpt_epoch{epoch}.pt'))
        
    with torch.no_grad():
        loss_val = 0
        dice_val = 0
        model.eval()
        val_steps = 0
        for moving, fixed, moving_labels, fixed_labels in val_gen:
            moving = resize_transform(moving)
            fixed = resize_transform(fixed)
            moving_labels = resize_labels_transform(moving_labels)
            fixed_labels = resize_labels_transform(fixed_labels)
            moving = moving.cuda().float()
            fixed = fixed.cuda().float()

            f_xy = model(moving, fixed)
            _, X_Y = transform(moving, f_xy.permute(0, 2, 3, 1))
            _, warped_labels = transform(moving_labels.cuda().float(), f_xy.permute(0, 2, 3, 1), mod='nearest')   
            for i in range(warped_labels.shape[0]):
                dice = utils.dice(warped_labels[i].detach().cpu().numpy().copy(), fixed_labels[i].detach().cpu().numpy().copy())
                dice_val += dice
            loss1 = loss_similarity(fixed, X_Y)
            loss5, _, _ = loss_smooth(f_xy)
        
            loss = loss1 + reg_param * loss5
            loss_val += loss.item()
            val_steps += 1
    
        scheduler.step()
        loss_val /= val_steps
        loss_train /= train_steps
        dice_val /= len(val)
        dice_progress.append(dice_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        writer.add_scalar("Loss/train", loss_train, epoch)
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("Dice/val", dice_val, epoch)
        print(f'Exp {n}: Epoch {epoch}, train loss: {(loss_train):5f}, val loss: {(loss_val):5f}, val dice: {(dice_val):5f}, learning rate: {scheduler.get_last_lr()[0]}')

writer.flush()
writer.close()

weights_path = os.path.join(exp_folder_name, 'weights.pth')
torch.save(model.state_dict(), weights_path)

loss_plot_name = os.path.join(exp_folder_name, 'dice_plot.png')
plt.plot(dice_progress, label='dice')
plt.xlabel('Epoch number')
plt.ylabel(f'Dice on validation')
plt.legend()

plt.savefig(loss_plot_name)