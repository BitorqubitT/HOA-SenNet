import torch as tc 
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
import numpy as np
from tqdm import tqdm

import cv2
import os
import sys
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

#!python -m pip install --no-index --find-links=/kaggle/input/pip-download-for-segmentation-models-pytorch segmentation-models-pytorch -q
import segmentation_models_pytorch as smp
from dotenv import load_dotenv

# Defining variables to specify model paths
model_path_i = 9  
model_path_i9 = 11

# Configuration class containing various model and training parameters
class CFG:
    # Model configuration
    model_name = 'Unet'
    backbone = 'se_resnext50_32x4d'
    in_chans = 1
    image_size = 1024
    input_size = 1024
    tile_size = image_size
    stride = tile_size // 4
    drop_egde_pixel = 0
    target_size = 1
    chopping_percentile = 1e-3

    # Fold and validation configuration
    valid_id = 1
    batch = 16
    th_percentile = 0.0014109
    axis_w = [0.328800989, 0.336629584, 0.334569427]
    axis_second_model = 0

    # Model paths for different folds
    model_path = [
        "/kaggle/input/sn-hoa-8e-5-27-rot0-5/se_resnext50_32x4d_26_loss0.10_score0.90_val_loss0.12_val_score0.88_midd_1024.pt",  # 8e-5-27-rot0-5
        "/kaggle/input/sn-hoa-8e-5-27-rot0-5/se_resnext50_32x4d_30_loss0.10_score0.90_val_loss0.13_val_score0.88_midd_1024.pt",
        "/kaggle/input/sennet-kidney-1-and-3/model_real_23.pt"  # 31 8e 05
    ]

# Custom model class definition
class CustomModel(nn.Module):
    def __init__(self, CFG, weight=None):
        super().__init__()
        
        self.CFG = CFG
        
        self.model = smp.Unet(
            encoder_name=CFG.backbone, 
            encoder_weights=weight,
            in_channels=CFG.in_chans,
            classes=CFG.target_size,
            activation=None,
        )
        
        # Setting batch size from the configuration
        self.batch = CFG.batch

    def forward_(self, image):
        # Forward pass through the model
        output = self.model(image)
        # Extracting the first channel from the output
        return output[:, 0]

    def forward(self, x: tc.Tensor):
        # Converting input tensor to float32
        x = x.to(tc.float32)
        
        # Normalizing the input tensor using a custom function 'norm_with_clip'
        x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
        
        # Interpolating the input tensor if input size is not equal to image size
        if CFG.input_size != CFG.image_size:
            x = nn.functional.interpolate(x, size=(CFG.input_size, CFG.input_size), mode='bilinear', align_corners=True)
        
        # Performing data augmentation by rotating the input tensor
        shape = x.shape
        x = [tc.rot90(x, k=i, dims=(-2, -1)) for i in range(4)]
        x = tc.cat(x, dim=0)
        
        # Using autocast for mixed precision training and no_grad to disable gradient computation
        with autocast():
            with tc.no_grad():
                # Forward pass for each rotated batch
                x = [self.forward_(x[i * self.batch:(i + 1) * self.batch]) for i in range(x.shape[0] // self.batch + 1)]
                # Concatenating the results along the batch dimension
                x = tc.cat(x, dim=0)
        
        x = x.sigmoid()
        x = x.reshape(4, shape[0], *shape[2:])
        
        # Rotating the tensor back to the original orientation
        x = [tc.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(4)]
        # Stacking along a new dimension and taking the mean
        x = tc.stack(x, dim=0).mean(0)
        
        # Interpolating the output tensor if input size is not equal to image size
        if CFG.input_size != CFG.image_size:
            x = nn.functional.interpolate(x[None], size=(CFG.image_size, CFG.image_size), mode='bilinear', align_corners=True)[0]
        
        return x

# Function to build the model
def build_model(weight=None):
    # Loading environment variables using dotenv
    load_dotenv()

    # Printing model name and backbone information
    print('model_name', CFG.model_name)
    print('backbone', CFG.backbone)

    # Creating an instance of the CustomModel with specified weight
    model = CustomModel(CFG, weight)

    # Moving the model to the GPU
    return model.cuda()

# Dataset class for loading images
class Data_loader(Dataset):
    def __init__(self, path, s="/images/"):
        self.paths = glob(path + f"{s}*.tif")
        self.paths.sort()
        self.bool = s == "/labels/"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)
        img = to_1024_1024(img, image_size=CFG.image_size)

        img = tc.from_numpy(img.copy())
        if self.bool:
            img = img.to(tc.bool)
        else:
            img = img.to(tc.uint8)
        return img

# Function to load data using Data_loader and perform normalization
def load_data(path, s):
    data_loader = Data_loader(path, s)
    data_loader = DataLoader(data_loader, batch_size=16, num_workers=2)
    data = []
    for x in tqdm(data_loader):
        data.append(x)
    x = tc.cat(data, dim=0)
    
    # Chopping values above and below a certain percentile
    TH = x.reshape(-1).numpy()
    index = -int(len(TH) * CFG.chopping_percentile)
    TH: int = np.partition(TH, index)[index]
    x[x > TH] = int(TH)
    
    TH = x.reshape(-1).numpy()
    index = -int(len(TH) * CFG.chopping_percentile)
    TH: int = np.partition(TH, -index)[-index]
    x[x < TH] = int(TH)
    
    return x

# Dataset class for pipeline processing
class Pipeline_Dataset(Dataset):
    def __init__(self, x, path):
        self.img_paths = glob(path + "/images/*")
        self.img_paths.sort()
        self.in_chan = CFG.in_chans
        z = tc.zeros(self.in_chan // 2, *x.shape[1:], dtype=x.dtype)
        self.x = tc.cat((z, x, z), dim=0)

    def __len__(self):
        return self.x.shape[0] - self.in_chan + 1

    def __getitem__(self, index):
        x = self.x[index:index + self.in_chan]
        return x, index

    def get_mark(self, index):
        id = self.img_paths[index].split("/")[-3:]
        id.pop(1)
        id = "_".join(id)
        return id[:-4]

    def get_marks(self):
        ids = []
        for index in range(len(self)):
            ids.append(self.get_mark(index))
        return ids


# Load both models and then combine the results
# Fix all these paths, they are different from my kaggle directory 

# Building and loading the 1024 x 1024 Image Model
model = build_model()  # Creating an instance of the model
model.load_state_dict(tc.load(CFG.model_path[model_path_i], "cpu"))  # Loading the pre-trained weights
model.eval()  # Setting the model to evaluation mode
model = DataParallel(model)  # Using DataParallel for parallel processing

# Building and loading the 512 x 512 Image Model
model9 = build_model()  # Creating an instance of the model
model9.load_state_dict(tc.load(CFG.model_path[model_path_i9], "cpu"))  # Loading the pre-trained weights
model9.eval()  # Setting the model to evaluation mode
model9 = DataParallel(model9)  # Using DataParallel for parallel processing

# Function to get segmentation outputs
def get_output(debug=False):
    outputs = []

    # Specify paths for testing
    if debug:
        paths = ["/kaggle/input/blood-vessel-segmentation/train/kidney_2"]
    else:
        paths = glob("/kaggle/input/blood-vessel-segmentation/test/*")

    outputs = [[], []]

    # Loop through each path
    for path in paths:
        # Load data and initialize labels
        x = load_data(path, "/images/")
        labels = tc.zeros_like(x, dtype=tc.uint8)
        mark = Pipeline_Dataset(x, path).get_marks()

        # Loop through each axis
        for axis in [0, 1, 2]:
            debug_count = 0

            # Rotate input data based on the current axis
            if axis == 0:
                x_ = x
                labels_ = labels
            elif axis == 1:
                x_ = x.permute(1, 2, 0)
                labels_ = labels.permute(1, 2, 0)
            elif axis == 2:
                x_ = x.permute(2, 0, 1)
                labels_ = labels.permute(2, 0, 1)

            # Skip if the input data is RGB and the axis is not 0
            if x.shape[0] == 3 and axis != 0:
                break

            # Create dataset and dataloader for processing
            dataset = Pipeline_Dataset(x_, path)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
            shape = dataset.x.shape[-2:]

            # Generate indices for processing tiles
            x1_list = np.arange(0, shape[0] + CFG.tile_size - CFG.tile_size + 1, CFG.stride)
            y1_list = np.arange(0, shape[1] + CFG.tile_size - CFG.tile_size + 1, CFG.stride)

            # Loop through each image in the dataloader
            for img, index in tqdm(dataloader):
                # img=(1,C,H,W)
                img = img.to("cuda:0")
                img = add_edge(img[0], CFG.tile_size // 2)[None]

                mask_pred = tc.zeros_like(img[:, 0], dtype=tc.float32, device=img.device)
                mask_count = tc.zeros_like(img[:, 0], dtype=tc.float32, device=img.device)

                indexs = []
                chip = []
                
                # Loop through each tile
                for y1 in y1_list:
                    for x1 in x1_list:
                        x2 = x1 + CFG.tile_size
                        y2 = y1 + CFG.tile_size
                        indexs.append([x1 + CFG.drop_egde_pixel, x2 - CFG.drop_egde_pixel,
                                       y1 + CFG.drop_egde_pixel, y2 - CFG.drop_egde_pixel])
                        chip.append(img[..., x1:x2, y1:y2])

                # Get predictions from the model
                y_preds = model.forward(tc.cat(chip)).to(device=0)

                # Weighted Ensemble
                if axis == CFG.axis_second_model:
                    y_preds = (0.90 * y_preds + 0.10 * model9.forward(tc.cat(chip)).to(device=0))

                # Adjust for drop_edge_pixel
                if CFG.drop_egde_pixel:
                    y_preds = y_preds[..., CFG.drop_egde_pixel:-CFG.drop_egde_pixel,
                                        CFG.drop_egde_pixel:-CFG.drop_egde_pixel]
                
                # Aggregate predictions over tiles
                for i, (x1, x2, y1, y2) in enumerate(indexs):
                    mask_pred[..., x1:x2, y1:y2] += y_preds[i]
                    mask_count[..., x1:x2, y1:y2] += 1

                mask_pred /= mask_count

                # Recover the region after processing
                mask_pred = mask_pred[..., CFG.tile_size // 2:-CFG.tile_size // 2, CFG.tile_size // 2:-CFG.tile_size // 2]

                # Update labels with the processed mask
                labels_[index] += (mask_pred[0] * 255 * CFG.axis_w[axis]).to(tc.uint8).cpu()

                # Display debug images if enabled
                if debug:
                    debug_count += 1
                    plt.subplot(121)
                    plt.imshow(img[0, CFG.in_chans // 2].cpu().detach().numpy())
                    plt.subplot(122)
                    plt.imshow(mask_pred[0].cpu().detach().numpy())
                    plt.show()
                    if debug_count > 3:
                        break
        
        # Append the labels and marks to the outputs list
        outputs[0].append(labels)
        outputs[1].extend(mark)
    
    return outputs

# Check if it's for submission based on the presence of test images
is_submit = len(glob("/kaggle/input/blood-vessel-segmentation/test/kidney_5/images/*.tif")) != 3

# Get segmentation output and associated ids
output, ids = get_output(not is_submit)

# Calculate threshold for binary predictions
TH = [x.flatten().numpy() for x in output]
TH = np.concatenate(TH)
index = -int(len(TH) * CFG.th_percentile)
TH: int = np.partition(TH, index)[index]
print(TH)

# Read an example image for visualization
img = cv2.imread("/kaggle/input/blood-vessel-segmentation/test/kidney_5/images/0001.tif", cv2.IMREAD_GRAYSCALE)

submission_df = []
debug_count = 0

# Loop through each prediction and generate RLE encoding
for index in range(len(ids)):
    id = ids[index]
    i = 0

    # Find the corresponding output based on the index
    for x in output:
        if index >= len(x):
            index -= len(x)
            i += 1
        else:
            break

    # Extract the binary mask based on the threshold
    mask_pred = (output[i][index] > TH).numpy()

    # Convert the binary mask to the original size
    mask_pred2 = to_original(mask_pred, img, image_size=1024)
    mask_pred = mask_pred2.copy()

    # Visualization (if not for submission)
    if not is_submit:
        plt.subplot(121)
        plt.imshow(mask_pred)
        plt.show()
        debug_count += 1
        if debug_count > 6:
            break

    # Encode the binary mask using Run-Length Encoding
    rle = rle_encode(mask_pred)

    # Append information to the submission dataframe
    submission_df.append(
        pd.DataFrame(data={
            'id': id,
            'rle': rle,
        }, index=[0])
    )

# Concatenate the submission dataframes and save to a CSV file
submission_df = pd.concat(submission_df)
submission_df.to_csv('submission.csv', index=False)
submission_df.head(6)