import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, Resize
import os
import re
import numpy as np
import pandas as pd
import cv2


def label_to_onehot(label):
    """
    Converts a segmentation label (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    colormap = [0, 1, 2]
    semantic_map = []
    label = label.numpy()
    for colour in colormap:
        equality = np.equal(label, colour)
        semantic_map.append(np.array(equality, dtype=float))
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def crop_transform(picture_size):
    return Compose([
        # Resize(picture_size),
        CenterCrop(picture_size), ])


def load_data(file_dir, picture_size=88):
    data_name = re.split('[/\\\]', file_dir)[-2]
    if data_name == 'SOC':
        label_name = {'BTR70': 0, 'BMP2': 1, 'BRDM_2': 2,'BTR_60': 3,'2S1': 4,'T72': 5, 'T62': 6,'ZIL131': 7,'D7': 8,
                       'ZSU_23_4': 9}
    path_list = []
    jpeg_list = []
    mask_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg':
                path_list.append(os.path.join(root, file))
    for jpeg_path in path_list:
        jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
        pic = crop_transform(picture_size)(torch.from_numpy(jpeg))
        jpeg_list.append(np.array(pic.div(pic.max())))
        mask_path = jpeg_path.replace('/SOC/TRAIN', '/SARBake').replace('jpeg', 'csv')
        mask = crop_transform(picture_size)(torch.from_numpy(pd.read_csv(mask_path, header=None).values))
        mask = label_to_onehot(mask)
        mask_list.append(mask)
        label_list.append(label_name[re.split('[/\\\.]', jpeg_path)[-4]])

    jpeg_list = np.array(jpeg_list)
    data = np.expand_dims(jpeg_list, axis=1)
    mask = np.transpose(np.array(mask_list), (0, 3, 1, 2))
    label = np.array(label_list)
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(data).type(torch.FloatTensor),
                                              torch.from_numpy(mask).type(torch.FloatTensor),
                                              torch.from_numpy(label).type(torch.LongTensor))
    return data_set, label_name


def load_test(file_dir, picture_size=88):
    data_name = re.split('[/\\\]', file_dir)[-2]
    if data_name == 'SOC':
        label_name = {'BTR70': 0, 'BMP2': 1, 'BRDM_2': 2,'BTR_60': 3,'2S1': 4,'T72': 5, 'T62': 6,'ZIL131': 7,'D7': 8,
                       'ZSU_23_4': 9}
    path_list = []
    jpeg_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg':
                path_list.append(os.path.join(root, file))
    for jpeg_path in path_list:
        jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
        pic = crop_transform(picture_size)(torch.from_numpy(jpeg))
        jpeg_list.append(np.array(pic.div(pic.max())))
        label_list.append(label_name[re.split('[/\\\.]', jpeg_path)[-4]])

    jpeg_list = np.array(jpeg_list)
    data = np.expand_dims(jpeg_list, axis=1)
    label = np.array(label_list)
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(data).type(torch.FloatTensor),
                                              torch.from_numpy(label).type(torch.LongTensor))
    return data_set


if __name__ == '__main__':
    train_all, label_name = load_data('../data/SOC/TRAIN')

