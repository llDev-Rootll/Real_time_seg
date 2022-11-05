import torch
from utils.utils import *
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2


LEARNING_RATE = 0.001
NUM_WORKERS = multiprocessing.cpu_count()//4
def main():
    train_transform = A.Compose([
    A.Resize(256, 512),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 
    # transform = None
    print("train")
    train_data = get_cityscapes_data(mode='fine', split='train', num_workers = NUM_WORKERS, batch_size = 8, transforms = train_transform, shuffle=True)
    val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 
    val_data = get_cityscapes_data(mode='fine', split='val', num_workers = NUM_WORKERS, batch_size = 8, transforms = val_transform, shuffle=True)
    test_data = get_cityscapes_data(mode='fine', split='test', num_workers = NUM_WORKERS, batch_size = 1, transforms = val_transform)
    print("Train data loaded successfully")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device", device, "found sucessfully!")
    
if __name__ == '__main__':
    main()