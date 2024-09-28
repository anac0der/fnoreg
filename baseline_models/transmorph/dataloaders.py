import numpy as np
from tqdm import tqdm
import os
import nibabel as nib
from utils import * 
 
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

class TrainOasis:
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = self.train_data.shape[0] * (self.train_data.shape[0] - 1)
        self.n = self.train_data.shape[0]
    
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        j = i % self.n if i % self.n != i // self.n else i % self.n + 1
        return np.expand_dims(self.train_data[i // self.n], axis=0), np.expand_dims(self.train_data[j], axis=0)

class ValidationOasis:
    def __init__(self, val_data, seg_labels):
        self.val_data = val_data
        self.seg_labels = seg_labels
        self.n = self.val_data.shape[0]
    
    def __len__(self):
        return (self.n - 1) * 2

    def __getitem__(self, i):
        if i < self.n - 1:
            ind1 = i
            ind2 = i + 1
        else:
            ind1 = (self.n - 1) * 2 - i
            ind2 = (self.n - 1) * 2 - i - 1

        return np.expand_dims(self.val_data[ind1], 0), np.expand_dims(self.val_data[ind2], 0), \
            np.expand_dims(self.seg_labels[ind1], 0), np.expand_dims(self.seg_labels[ind2], 0)

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