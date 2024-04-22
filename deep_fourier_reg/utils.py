import cv2
import numpy as np
from tqdm import tqdm
import torch.utils.data as Data
import pandas as pd
import os
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import metrics
import nibabel as nib
from matplotlib.collections import LineCollection
import flow_vis
from prettytable import PrettyTable

def vis(img1, img2):
    def autocontrast(img):
        return (255 / (img.max() - img.min())) * (img - img.min())

    result1 = np.zeros((img1.shape[0], img1.shape[1], 3))
    result2 = np.zeros((img1.shape[0], img1.shape[1], 3))

    img1[img1 > 10] += 30
    img1[img1 > 255] = 255
    img2[img2 > 10] += 30
    img2[img2 > 255] = 255

    result1[:, :, 0] = img1
    result1[:, :, 2] = img1
    result2[:, :, 1] = img2
    result2[:, :, 2] = img2
    result1 = autocontrast(result1)
    result2 = autocontrast(result2)
    return result1, result2, autocontrast(result1 + result2)

def plot_landmarks(img, landmarks, save_path='default.png', flow=False, flow_values=None, fixed=False, fixed_landmarks=None):
    plt.figure(figsize=(18, 12))
    plt.tick_params (left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)
    plt.imshow(img, cmap="gist_gray")
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker="+", color='red')
    if flow:
        new_l = landmarks + flow_values
        plt.scatter(new_l[:, 0], new_l[:, 1], marker="*", color='blue')
        if fixed:
            plt.scatter(fixed_landmarks[:, 0], fixed_landmarks[:, 1], marker="x", color='green')
    plt.savefig(save_path)
    plt.close()

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

def plotter(case_id, source, target, transformed_target, flow):

    flow_color = flow_vis.flow_to_color(flow.transpose(1, 2, 0) * 128).astype('float32') / 255

    target_yellow = np.stack((target, target, np.zeros_like(target)), axis=-1)
    source_magenta = np.stack((source, np.zeros_like(source), source), axis=-1)
    transformed_target_yellow = np.stack((transformed_target, transformed_target, np.zeros_like(transformed_target)), axis=-1)
    transformed_target_yellow = 0.5 * transformed_target_yellow + 0.5 * source_magenta


    cv2.putText(img=source_magenta, text=case_id + ': fixed', org=(50, 250),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                color=(1, 1, 1), thickness=1, lineType=cv2.LINE_AA)
    
    cv2.putText(img=target_yellow, text=case_id + ': moving', org=(50, 250),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, 
                color=(1, 1, 1), thickness=1, lineType=cv2.LINE_AA)

    cv2.putText(img=transformed_target_yellow, text=case_id + ': moved', org=(50, 250),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                color=(1, 1, 1), thickness=1, lineType=cv2.LINE_AA)
    
    cv2.putText(img=flow_color, text=case_id + ': flow', org=(50, 250),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                color=(1, 1, 1), thickness=1, lineType=cv2.LINE_AA)

    separator = int(min(source.shape[:2]) * 0.05)
    vertical_line = np.ones((source.shape[0], separator, 3))
    horisontal_line = np.ones((separator, 2*source.shape[1] + separator , 3))
    
    up = np.hstack((source_magenta, vertical_line, target_yellow))
    down = np.hstack((transformed_target_yellow, vertical_line, flow_color))
    
    result = np.vstack((up, horisontal_line, down))

    return result
 
def evaluate(model, test_gen, amount, patch_size):
    print('Evaluating model...')
    mrtre_after_reg = []
    for i in tqdm(range(amount)):
        val = next(test_gen)
        inp = val[0]
        img, field = model.predict(inp, verbose=0)
        fixed = inp[1][0, :, :, 0]
        moving = inp[0][0, :, :, 0]
        moved = img[0].reshape((patch_size, patch_size))
        landmarks = val[2]
        fixed_landmarks = landmarks[0][0]
        if fixed_landmarks.shape[0] < 5:
            continue
        moving_landmarks = landmarks[1][0]
        moved_landmarks = transform_landmarks(moving_landmarks, field.squeeze())
        image_diagonal = np.sqrt(fixed.shape[0] ** 2 + fixed.shape[1] ** 2)
        mrtre_after_reg.append(np.median(metrics.rTRE(moved_landmarks, fixed_landmarks, image_diagonal)))
    return np.array(mrtre_after_reg).mean()

def load_landmarks(landmarks_path):
    landmarks = pd.read_csv(landmarks_path)
    landmarks = landmarks.to_numpy()[:, 1:]
    return landmarks

def shift_landmarks(landmarks, shift_vector):
    new_landmarks = landmarks.copy()
    new_landmarks[:, 0] += shift_vector[0]
    new_landmarks[:, 1] += shift_vector[1]
    return new_landmarks

def pad_landmarks(landmarks, old_shape, new_shape):
    new_landmarks = landmarks.copy()
    new_landmarks[:, 0] += int(np.floor((new_shape[1] - old_shape[1])/2))
    new_landmarks[:, 1] += int(np.floor((new_shape[0] - old_shape[0])/2))
    return new_landmarks

def resample_landmarks(landmarks, resample_ratio):
    new_landmarks = landmarks / resample_ratio
    return new_landmarks

def resample_pad_landmarks(landmarks, ratio, before_pad_shape, after_pad_shape):
    res_landmarks = resample_landmarks(landmarks, ratio)
    return pad_landmarks(res_landmarks, before_pad_shape, after_pad_shape)

def filter_landmarks(fixed_landmarks, moving_landmarks, center, patch_size):
    indexes_fixed = landmarks_in_patch(fixed_landmarks, center, patch_size)
    indexes_moving = landmarks_in_patch(moving_landmarks, center, patch_size)
    indexes_intersect = list(set(indexes_fixed) & set(indexes_moving))
    new_fixed_landmarks = fixed_landmarks[indexes_intersect, :]
    new_moving_landmarks = moving_landmarks[indexes_intersect, :]
    shift_vector = np.array([center[0] - patch_size // 2, center[1] - patch_size // 2])
    return shift_landmarks(new_fixed_landmarks, -shift_vector), shift_landmarks(new_moving_landmarks, -shift_vector)

def landmarks_in_patch(landmarks, center, patch_size):
    indexes = []
    x = center[0]
    y = center[1]
    for i in range(landmarks.shape[0]):
        if x - patch_size // 2 <= landmarks[i, 0] < x + patch_size // 2 and y - patch_size // 2 <= landmarks[i, 1] < y + patch_size // 2:
            indexes.append(i)
    return indexes
    
def transform_landmarks(landmarks, displacement_field):
    u_x = displacement_field[1, :, :]
    u_y = displacement_field[0, :, :]
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(u_x, [landmarks_y, landmarks_x], mode='nearest')
    uy = nd.map_coordinates(u_y, [landmarks_y, landmarks_x], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)
    flow_values = np.stack([ux, uy], axis=1)
    return new_landmarks, flow_values

def dice(pred1, truth1):
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    dice_list=[]
    for k in mask_value4[1:]:
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        # print(intersection)
        dice_list.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_list)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

