import numpy as np
from torchvision import transforms
import torch
from tqdm import tqdm
from dataset.CityscapesDataset import CityscapesDataset
from utils.utils import *
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import time

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

colormap = np.zeros((19, 3), dtype=np.uint8)
colormap[0] = [128, 64, 128]
colormap[1] = [244, 35, 232]
colormap[2] = [70, 70, 70]
colormap[3] = [102, 102, 156]
colormap[4] = [190, 153, 153]
colormap[5] = [153, 153, 153]
colormap[6] = [250, 170, 30]
colormap[7] = [220, 220, 0]
colormap[8] = [107, 142, 35]
colormap[9] = [152, 251, 152]
colormap[10] = [70, 130, 180]
colormap[11] = [220, 20, 60]
colormap[12] = [255, 0, 0]
colormap[13] = [0, 0, 142]
colormap[14] = [0, 0, 70]
colormap[15] = [0, 60, 100]
colormap[16] = [0, 80, 100]
colormap[17] = [0, 0, 230]
colormap[18] = [119, 11, 32]

def get_cityscapes_data(
    mode,
    split,
    root_dir='Cityscapes',
    target_type="semantic",
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,
    num_workers=2

):
    data = CityscapesDataset(
        mode=mode, split=split, target_type=target_type,transform=transforms, root_dir=root_dir, eval=eval)

    data_loaded = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    return data_loaded

