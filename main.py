import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from model.UNet_backbone import Unet
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.loss import FocalLoss
import train

# from torchsummary import summary
torch.manual_seed(0)
LEARNING_RATE = 0.001
NUM_WORKERS = multiprocessing.cpu_count()//4
BACKBONE = "densenet169"
EPOCHS = 100

EXP_NAME = BACKBONE + "_" + str(time.time())
# EXP_NAME = "log_dir/densenet169_1670792529.2218642"
SAVE_PATH = os.path.join("log_dir",EXP_NAME)
def main():
    train_transform = A.Compose([
    # A.RandomSizedCrop((256 - 32, 512 - 64), 256, 512, p=1),
    # A.RandomRotate90(p=0.3),
    A.Resize(256, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 
    val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 

    train_loader = get_cityscapes_data(mode='fine', split='train', num_workers = NUM_WORKERS, batch_size = 4, transforms = train_transform, shuffle=True)
    val_loader = get_cityscapes_data(mode='fine', split='val', num_workers = NUM_WORKERS, batch_size = 1, transforms = val_transform, shuffle=True)
    # test_loader = get_cityscapes_data(mode='fine', split='test', num_workers = NUM_WORKERS, batch_size = 1, transforms = val_transform)

    # for img, label, x_p, y_p in tqdm(train_loader):
    #     print(x_p)
    #     print(y_p)
    #     unique_ids = np.unique(label.numpy().astype(float))
    #     if unique_ids.size>2:
    #         # if unique_ids[0]==255:
    #         print(np.unique(label.numpy().astype(float)))
    #         # print(label.numpy().astype(float).dtype)
    #         plt.figure()
    #         plt.imshow(label[0].numpy().astype(float))
    #         plt.figure()
    #         plt.imshow(img[0].permute(1,2,0).numpy())
    #         plt.show()
    # exit(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device", device, "found sucessfully!")
    model = Unet(backbone_name=BACKBONE).to(device)
    print("Model loaded")
    # ckpt = torch.load("log_dir/densenet169_1670792529.2218642/densenet169_40.pt")
    # model.load_state_dict(ckpt["model_state_dict"])
    
    # optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.2, patience=2)
    # criterion = nn.CrossEntropyLoss(ignore_index = 255) 
    alpha = torch.tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]).to(device)
    criterion = FocalLoss(alpha = alpha, gamma = 2, ignore_index = 255)
    print("Optimizer and Loss defined")

    # print("############### Start Training ################")
    train.train_model(num_epochs=EPOCHS, model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_function=criterion, scheduler=scheduler, save_path = SAVE_PATH)
    # evaluate_(test_data, current_model, "finalUNet_res3450.pt", "test_results")
    # rt_vid_path = "stuttgart_01"
    # real_time_segmentation(model, device, "log_dir/densenet169_1670885130.7382257/densenet169_40.pt", rt_vid_path, transform = val_transform)
    # evaluate_(data_loader=val_loader, model=model, path="log_dir/densenet169_1670885130.7382257/densenet169_40.pt", save_path="saved_images")
if __name__ == '__main__':
    main()