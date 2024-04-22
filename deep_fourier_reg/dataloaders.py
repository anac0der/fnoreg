import cv2
import numpy as np
from tqdm import tqdm
import torch.utils.data as Data
import pandas as pd
import os
import nibabel as nib
import memory_profiler
from utils import *

class Datagen(Data.Dataset):
    def __init__(self, n, fixed, moving, params, seed=1147, landmarks=False, fixed_landmarks_src=None, moving_landmarks_src=None):
        self.n = n
        self.rng = np.random.default_rng(seed=seed)
        self.downsample_size = params['downsample_size'][0]
        self.patch_size = params['patch_size'][0]
        self.fixed = fixed
        self.moving = moving
        self.landmarks = landmarks
        if self.landmarks:
            self.fixed_landmarks_src = fixed_landmarks_src
            self.moving_landmarks_src = moving_landmarks_src

    def __len__(self):
        return self.n
    
    def __getitem__(self, index=0):
        imgs_index = self.rng.integers(0, self.fixed.shape[0], size=1)
        patch_indexes = self.rng.integers(self.patch_size, self.downsample_size - self.patch_size, size=(1, 2))
        y = patch_indexes[0, 0]
        x = patch_indexes[0, 1]
        f = self.fixed[imgs_index, y - self.patch_size // 2:y + self.patch_size // 2, x - self.patch_size // 2:x + self.patch_size // 2].reshape((1, self.patch_size, self.patch_size))
        m = self.moving[imgs_index, y - self.patch_size // 2:y + self.patch_size // 2, x - self.patch_size // 2:x + self.patch_size // 2].reshape((1, self.patch_size, self.patch_size))
        while np.sum(np.where(f > 0.001, 1, 0)) < self.patch_size ** 2 / 2:
            patch_indexes_i = self.rng.integers(self.patch_size, self.downsample_size - self.patch_size, size=(1, 2))
            y = patch_indexes_i[0, 0]
            x = patch_indexes_i[0, 1]
            f = self.fixed[imgs_index, y - self.patch_size // 2:y + self.patch_size // 2, x - self.patch_size // 2:x + self.patch_size // 2].reshape((1, self.patch_size, self.patch_size))
            m = self.moving[imgs_index, y - self.patch_size // 2:y + self.patch_size // 2, x - self.patch_size // 2:x + self.patch_size // 2].reshape((1, self.patch_size, self.patch_size))
        if self.landmarks:
            center = [x, y]
            # print(self.fixed_landmarks_src)
            fixed_curr_landmarks = self.fixed_landmarks_src[imgs_index][0]
            # print(fixed_curr_landmarks.shape)
            moving_curr_landmarks = self.moving_landmarks_src[imgs_index][0]
            fixed_filtered_landmarks, moving_filtered_landmarks = filter_landmarks(fixed_curr_landmarks, moving_curr_landmarks, center, self.patch_size)
        if not self.landmarks:
            return m, f
        else:        
            return m, f, moving_filtered_landmarks, fixed_filtered_landmarks 

class ArrayWrapper:
    def __init__(self, fixed, moving, landmarks=False, fixed_landmarks_src=None, moving_landmarks_src=None):
        self.fixed = fixed
        self.moving = moving
        self.landmarks = landmarks
        if self.landmarks:
            self.fixed_landmarks_src = fixed_landmarks_src
            self.moving_landmarks_src = moving_landmarks_src
        
    def __getitem__(self, i):
        if not self.landmarks:
            return np.expand_dims(self.moving[i], axis=0), np.expand_dims(self.fixed[i], axis=0)
        else:
            return np.expand_dims(self.moving[i], axis=0), np.expand_dims(self.fixed[i], axis=0), \
                    self.moving_landmarks_src[i], self.fixed_landmarks_src[i]

    def __len__(self):
        return self.fixed.shape[0]
    
def preprocessing(image, downsample_size, landmarks=False, landmarks_src=None):
    height, width = image.shape
    if height > width:
        resample_ratio = height / downsample_size
        dim = int((width / height) * downsample_size)
        img = cv2.resize(image, (dim, downsample_size))
        pad_size = (downsample_size - dim) // 2
        res_img = np.pad(img, ((0, 0), (pad_size, downsample_size - dim - pad_size)))
        if not landmarks:
            return res_img, None
        else:
            return res_img, resample_pad_landmarks(landmarks_src, resample_ratio, img.shape, res_img.shape)
    
    else:
        resample_ratio = width / downsample_size
        dim = int((height / width) * downsample_size)
        img = cv2.resize(image, (downsample_size, dim))
        pad_size = (downsample_size - dim) // 2
        res_img = np.pad(img, ((pad_size, downsample_size - dim - pad_size), (0, 0)))
        if not landmarks:
            return res_img, None
        else:
            return res_img, resample_pad_landmarks(landmarks_src, resample_ratio, img.shape, res_img.shape)

def load_dataset(dataset_size, downsample_size, path, csv_path, landmarks=False, landmarks_path=None): 
    csv = pd.read_csv(csv_path)
    status_counts = csv['status'].value_counts()
    train_cnt = status_counts['training']
    test_cnt = status_counts['evaluation']

    fixed_images_train = np.empty((train_cnt, downsample_size, downsample_size))
    moving_images_train = np.empty((train_cnt, downsample_size, downsample_size))
    fixed_images_test = np.empty((test_cnt, downsample_size, downsample_size))
    moving_images_test = np.empty((test_cnt, downsample_size, downsample_size))
    fixed_images_test_landmarks = np.empty(test_cnt, dtype=object)
    moving_images_test_landmarks = np.empty(test_cnt, dtype=object)
    print("Loading dataset...")
    j_train = 0
    j_test = 0
    for i in tqdm(range(dataset_size)):
        f = cv2.imread(path + f"/{i}/target.png", cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(path + f"/{i}/transformed_source.png", cv2.IMREAD_GRAYSCALE) 
        if csv['status'][i] == 'training':
            prepr_fixed, _ =  preprocessing(f, downsample_size)
            fixed_images_train[j_train] = prepr_fixed.astype('float32') / 255
            prepr_moving, _ =  preprocessing(m, downsample_size)
            moving_images_train[j_train] = prepr_moving.astype('float32') / 255
            j_train += 1
        else:
            if landmarks:
                fixed_landmarks = load_landmarks(os.path.join(landmarks_path, str(i), 'target_landmarks.csv'))
                moving_landmarks = load_landmarks(os.path.join(landmarks_path, str(i), 'transformed_source_landmarks.csv'))
            else:
                fixed_landmarks, moving_landmarks = None, None
            prepr_fixed_test, prepr_fixed_landmarks = preprocessing(f, downsample_size, landmarks=landmarks, landmarks_src=fixed_landmarks)
            fixed_images_test[j_test] = prepr_fixed_test.astype('float32') / 255
            prepr_moving_test, prepr_moving_landmarks = preprocessing(m, downsample_size, landmarks=landmarks, landmarks_src=moving_landmarks)
            moving_images_test[j_test] = prepr_moving_test.astype('float32') / 255
            if landmarks:
                fixed_images_test_landmarks[j_test] = prepr_fixed_landmarks
                moving_images_test_landmarks[j_test] = prepr_moving_landmarks
            j_test += 1
    return fixed_images_train, fixed_images_test, moving_images_train, moving_images_test, \
        fixed_images_test_landmarks, moving_images_test_landmarks

def load_oasis(dataset_path, folders_list):
    data_len = len(folders_list)
    print(data_len)
    dataset = np.zeros((data_len, 160, 192))
    seg_labels_data = np.zeros((data_len, 160, 192))
    i = 0
    for folder in tqdm(folders_list, ncols=100):
        path = os.path.join(dataset_path, folder)
        image_path = os.path.join(path, 'slice_norm.nii.gz')
        seg_labels_path = os.path.join(path, 'slice_seg24.nii.gz')
        nim1 = nib.load(image_path)
        image = nim1.get_fdata()[:,:,0]
        # print(image)
        image = np.array(image, dtype='float32')
        # print(image)

        nim2 = nib.load(seg_labels_path)
        seg_labels = nim2.get_fdata()[:,:,0]
        seg_labels = np.array(seg_labels, dtype='float32')

        dataset[i, ...] = image.reshape((160, 192))
        seg_labels_data[i, ...] = seg_labels.reshape((160, 192))
        i += 1

    return dataset, seg_labels_data

def load_oasis3d(dataset_path, folders_list):
    data_len = len(folders_list)
    dataset = np.zeros((data_len, 160, 192, 224))
    seg_labels_data = np.zeros((data_len, 160, 192, 224))
    i = 0
    for folder in tqdm(folders_list, ncols=100):
        path = os.path.join(dataset_path, folder)
        image_path = os.path.join(path, 'aligned_norm.nii.gz')
        seg_labels_path = os.path.join(path, 'aligned_seg35.nii.gz')
        nim1 = nib.load(image_path)
        image = nim1.get_fdata()
        image = np.array(image, dtype='float32')

        nim2 = nib.load(seg_labels_path)
        seg_labels = nim2.get_fdata()
        seg_labels = np.array(seg_labels, dtype='float32')

        dataset[i, ...] = image.reshape((160, 192, 224))
        seg_labels_data[i, ...] = seg_labels.reshape((160, 192, 224))
        i += 1
    
    return dataset, seg_labels_data

class TrainOasis2d:
    def __init__(self, train_indices, dataset_path, folders):
        self.train_indices = train_indices
        self.dataset_path = dataset_path
        folders = np.array(folders, dtype=object)
        self.folders = folders[train_indices]
        self.n = self.folders.shape[0]
        self.length = self.n * (self.n - 1)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        j = i % self.n if i % self.n != i // self.n else i % self.n + 1
        return self.load_pair(i // self.n, j)
    
    def load_image(self, i):
        path = os.path.join(self.dataset_path, self.folders[i])
        image_path = os.path.join(path, 'slice_norm.nii.gz')
        nim1 = nib.load(image_path)
        image1 = nim1.get_fdata()[:, :, 0]
        image1 = np.array(image1, dtype='float32')

        return image1
    
    def load_pair(self, i, j):
        return np.expand_dims(self.load_image(i), 0), np.expand_dims(self.load_image(j), 0)

class ValidationOasis2d:
    def __init__(self, val_indices, dataset_path, folders):
        self.val_indices = val_indices
        self.dataset_path = dataset_path
        folders = np.array(folders, dtype=object)
        self.folders = folders[val_indices]
        self.n = self.folders.shape[0]
        self.length = self.n * (self.n - 1)
    
    def __len__(self):
        return (self.n - 1) * 2

    def __getitem__(self, i):
        if i < self.n - 1:
            ind1 = i
            ind2 = i + 1
        else:
            ind1 = (self.n - 1) * 2 - i
            ind2 = (self.n - 1) * 2 - i - 1

        return self.load_pair(ind1, ind2)  
    
    def load_image(self, i):
        path = os.path.join(self.dataset_path, self.folders[i])
        image_path = os.path.join(path, 'slice_norm.nii.gz')
        nim1 = nib.load(image_path)
        image1 = nim1.get_fdata()[:, :, 0]
        image1 = np.array(image1, dtype='float32')
        return image1

    def load_label(self, i):
        path = os.path.join(self.dataset_path, self.folders[i])
        image_path = os.path.join(path, 'slice_seg24.nii.gz')
        nim1 = nib.load(image_path)
        label1 = nim1.get_fdata()[:, :, 0]
        label1 = np.array(label1, dtype='float32')
        return label1

    def load_pair(self, i, j):
        return np.expand_dims(self.load_image(i), 0), np.expand_dims(self.load_image(j), 0), np.expand_dims(self.load_label(i), 0), np.expand_dims(self.load_label(j), 0)
    
class TrainOasis3d:
    def __init__(self, val_indices, dataset_path, folders):
        self.val_indices = val_indices
        self.dataset_path = dataset_path
        folders = np.array(folders, dtype=object)
        self.folders = folders[val_indices]
        self.n = self.folders.shape[0]
    
    def __len__(self):
        return (self.n - 1) * 2

    def __getitem__(self, i):
        if i < self.n - 1:
            ind1 = i
            ind2 = i + 1
        else:
            ind1 = (self.n - 1) * 2 - i
            ind2 = (self.n - 1) * 2 - i - 1

        return self.load_pair(ind1, ind2)  
    
    def load_image(self, i):
        path = os.path.join(self.dataset_path, self.folders[i])
        image_path = os.path.join(path, 'aligned_norm.nii.gz')
        nim1 = nib.load(image_path)
        image1 = nim1.get_fdata()
        image1 = np.array(image1, dtype='float32')
        return image1

    def load_label(self, i):
        path = os.path.join(self.dataset_path, self.folders[i])
        image_path = os.path.join(path, 'aligned_seg35.nii.gz')
        nim1 = nib.load(image_path)
        label1 = nim1.get_fdata()
        label1 = np.array(label1, dtype='float32')
        return label1

    def load_pair(self, i, j):
        return np.expand_dims(self.load_image(i), 0), np.expand_dims(self.load_image(j), 0), np.expand_dims(self.load_label(i), 0), np.expand_dims(self.load_label(j), 0)

class ValidationOasis3d:
    def __init__(self, val_indices, dataset_path, folders):
        self.val_indices = val_indices
        self.dataset_path = dataset_path
        folders = np.array(folders, dtype=object)
        self.folders = folders[val_indices]
        self.n = self.folders.shape[0]
    
    def __len__(self):
        return (self.n - 1)

    def __getitem__(self, i):
        ind1 = i
        ind2 = i + 1
        return self.load_pair(ind1, ind2)  
    
    def load_image(self, i):
        path = os.path.join(self.dataset_path, self.folders[i])
        image_path = os.path.join(path, 'aligned_norm.nii.gz')
        nim1 = nib.load(image_path)
        image1 = nim1.get_fdata()
        image1 = np.array(image1, dtype='float32')
        return image1

    def load_label(self, i):
        path = os.path.join(self.dataset_path, self.folders[i])
        image_path = os.path.join(path, 'aligned_seg35.nii.gz')
        nim1 = nib.load(image_path)
        label1 = nim1.get_fdata()
        label1 = np.array(label1, dtype='float32')
        return label1

    def load_pair(self, i, j):
        return np.expand_dims(self.load_image(i), 0), np.expand_dims(self.load_image(j), 0), np.expand_dims(self.load_label(i), 0), np.expand_dims(self.load_label(j), 0)