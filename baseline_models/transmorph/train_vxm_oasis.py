import os
import dataloaders
import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import re
import pandas as pd
from utils import dice, count_parameters
from Baseline_registration_models.VoxelMorph.models import VxmDense_1, VxmDense_huge
import Baseline_registration_models.VoxelMorph.utils as utils
import Baseline_registration_models.VoxelMorph.losses as losses
import torchvision.transforms.v2 as transforms

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int,  default=0, help='GPU number')
parser.add_argument('--config_file', type=str,  default='params_vxm.json', help='JSON config file name')
parser.add_argument('--exp_num', type=int,  default=-1, help='Exp number to retrain, optional')
parser.add_argument('--ckpt_epoch', type=int, default=-1, help='Checkpoint epoch number')
parser.add_argument('--size', type=int, default=160, help='Size of smaller dim of data sample, optional')
args = parser.parse_args()

params = pd.read_json(args.config_file)
EXP_PATH = params['exp_path'][0]
OASIS_FOLDERS_PATH = params['oasis_folders_path'][0]
OASIS_PATH = params['oasis_path'][0]

oasis_folders = []
with open(OASIS_FOLDERS_PATH, 'r') as f:
    for line in f:
        oasis_folders.append(line.strip('\n'))

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
device = torch.device("cuda")
print(torch.cuda.current_device())

print('Loading OASIS dataset...')
oasis_data, oasis_labels = dataloaders.load_oasis(OASIS_PATH, oasis_folders)
oasis_train = oasis_data[:201]
oasis_val = oasis_data[201:213]
oasis_val_labels = oasis_labels[201:213]

if args.exp_num < 0:
    model_name = params["model_name"][0]
    train_config = params['train_config'][0]
else:
    exp_folder_name = os.path.join(EXP_PATH, F'oasis_exp{args.exp_num}')
    with open(os.path.join(exp_folder_name, 'metadata.json'), "r") as f:
        exp_metadata_str = f.read()
    exp_metadata = json.loads(exp_metadata_str)
    model_name = exp_metadata['model_name']
    train_config = exp_metadata['train_config']

inshape = (args.size, int(args.size * (192 / 160)))
if model_name == 'vxm-1':
    model = VxmDense_1(inshape=inshape).cuda()
elif model_name == 'vxm-huge':
    model = VxmDense_huge(inshape=inshape).cuda()
else:
    raise Exception('Incorrect model name!')

count_parameters(model) 

reg_model = utils.register_model(img_size=inshape, mode='nearest')
reg_model.cuda()

seed = 2002
bs = train_config['batch_size']
train = dataloaders.TrainOasis(oasis_train)
val = dataloaders.ValidationOasis(oasis_val, oasis_val_labels)
train_gen = Data.DataLoader(dataset=train, batch_size=bs, shuffle=True, num_workers=4)
val_gen = Data.DataLoader(dataset=val, batch_size=bs, shuffle=False, num_workers=2)
reg_param = train_config['reg_param']

loss_sim = torch.nn.MSELoss(reduction='mean')
loss_smooth = losses.Grad(penalty='l2')

optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
if train_config['lr_scheduler']:
    name = train_config['lr_scheduler']
    sch_params = train_config['lr_scheduler_params'][name]
    if name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sch_params)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10 ** 8, gamma=1)

exp_config = params['exp_config'][0]
if args.exp_num < 0:
    exp_dirs = []
    for _, dirs, _ in os.walk(EXP_PATH):
        for dir in dirs:
            if re.search('oasis_exp.', dir):
                dir = re.sub('[^0-9]', '0', dir)
                exp_dirs.append(int(dir))
    if len(exp_dirs) == 0:
        n = 0
    else:
        n = sorted(exp_dirs)[-1] + 1
else:
    n = args.exp_num

exp_folder_name = os.path.join(EXP_PATH, f'oasis_exp{n}')
try:
    os.mkdir(exp_folder_name)
except Exception:
    print(f'Folder for experiment {n} is already exists!')

exp_description = exp_config['description']
print(f'Experiment {n}: {exp_description}')
exp_metadata = dict()
exp_metadata['description'] = exp_description
exp_metadata['start time'] = str(datetime.now())
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

train_losses = []
val_losses = []
dice_progress = []
writer = SummaryWriter(log_dir=os.path.join(exp_folder_name, "./logs"))
resize_transform = transforms.Resize(size=inshape)
resize_labels_transform = transforms.Resize(size=inshape, interpolation=transforms.InterpolationMode.NEAREST)
for epoch in range(init_epoch, train_config['epochs']):
    loss_train = 0
    model.train()
    print(f'Epoch {epoch} started...')
    adjust_learning_rate(optimizer, epoch, train_config['epochs'], train_config['lr'])
    train_steps = 0 
    for moving, fixed in tqdm(train_gen, ncols=100):
        moving = resize_transform(moving)
        fixed = resize_transform(fixed)
        moving = moving.cuda().float()
        fixed = fixed.cuda().float()
        x = torch.cat([moving, fixed], 1) 
        X_Y, f_xy = model(x)
        loss1 = loss_sim(fixed, X_Y)
        loss5 = loss_smooth(f_xy)

        loss = loss1 + reg_param * loss5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss
        train_steps += 1
    
    if epoch % train_config['ckpt_freq'] == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
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

            x = torch.cat([moving, fixed], 1)
            X_Y, f_xy = model(x)
            warped_labels = reg_model([moving_labels.cuda().float(), f_xy.float()])   
            for i in range(warped_labels.shape[0]):
                dice_score = dice(warped_labels[i].detach().long().cpu().numpy().copy(), fixed_labels[i].detach().long().cpu().numpy().copy())
                dice_val += dice_score
            loss1 = loss_sim(fixed, X_Y)
            loss5 = loss_smooth(f_xy)

            loss = loss1 + reg_param * loss5
            loss_val += loss
            val_steps += 1
    
        scheduler.step()
        loss_val = loss_val.item()
        loss_val /= val_steps
        loss_train = loss_train.item()
        loss_train /= train_steps
        dice_val /= len(val)
        dice_progress.append(dice_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        writer.add_scalar("Loss/train", loss_train, epoch)
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("Dice/val", dice_val, epoch)
        print(f'Epoch {epoch}, train loss: {(loss_train):5f}, val loss: {(loss_val):5f}, val dice: {(dice_val):5f}, learning rate: {scheduler.get_last_lr()[0]}')

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