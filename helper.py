import torch as tc
import numpy as np

# Function to resize image back to original size
def to_original(im_after, img, image_size=1024):
    top_ = 0
    left_ = 0
    
    # Calculate padding for top
    if im_after.shape[0] > img.shape[0]:
        top_ = (image_size - img.shape[0]) // 2
    
    # Calculate padding for left
    if im_after.shape[1] > img.shape[1]:
        left_ = (image_size - img.shape[1]) // 2
    
    # Extract the region of interest from the resized image
    if (top_ > 0) or (left_ > 0):
        img_result = im_after[top_: img.shape[0] + top_, left_: img.shape[1] + left_]
    else:
        img_result = im_after
    return img_result

# Function to encode a binary mask using Run-Length Encoding (RLE)
def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

# Function for min-max normalization of a Pytc tensor
def min_max_normalization(x: tc.Tensor) -> tc.Tensor:
    shape = x.shape
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    min_ = x.min(dim=-1, keepdim=True)[0]
    max_ = x.max(dim=-1, keepdim=True)[0]
    if min_.mean() == 0 and max_.mean() == 1:
        return x.reshape(shape)

    x = (x - min_) / (max_ - min_ + 1e-9)
    return x.reshape(shape)

# Function for normalization with clipping of a Pytc tensor
def norm_with_clip(x: tc.Tensor, smooth=1e-5):
    dim = list(range(1, x.ndim))
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + smooth)
    x[x > 5] = (x[x > 5] - 5) * 1e-3 + 5
    x[x < -3] = (x[x < -3] + 3) * 1e-3 - 3
    return x

# Function to add an edge to an image tensor
def add_edge(x: tc.Tensor, edge: int):
    # x=(C,H,W)
    # output=(C,H+2*edge,W+2*edge)
    mean_ = int(x.to(tc.float32).mean())
    x = tc.cat([x, tc.ones([x.shape[0], edge, x.shape[2]], dtype=x.dtype, device=x.device) * mean_], dim=1)
    x = tc.cat([x, tc.ones([x.shape[0], x.shape[1], edge], dtype=x.dtype, device=x.device) * mean_], dim=2)
    x = tc.cat([tc.ones([x.shape[0], edge, x.shape[2]], dtype=x.dtype, device=x.device) * mean_, x], dim=1)
    x = tc.cat([tc.ones([x.shape[0], x.shape[1], edge], dtype=x.dtype, device=x.device) * mean_, x], dim=2)
    return x

# Function to resize image to 1024x1024 with rotation
def to_1024_1024(img, CFG, image_size=1024):
    if image_size > img.shape[1]:
        # Rotate image 90 degrees
        img = np.rot90(img)
        
        # Calculate padding for top and bottom
        start1 = (CFG.image_size - img.shape[0]) // 2
        top = img[0: start1, 0: img.shape[1]]
        bottom = img[img.shape[0] - start1: img.shape[0], 0: img.shape[1]]
        
        # Concatenate top, rotated image, and bottom
        img_result = np.concatenate((top, img, bottom), axis=0)
        
        # Rotate image back to the original orientation
        img_result = np.rot90(img_result)
        img_result = np.rot90(img_result)
        img_result = np.rot90(img_result)
    else:
        img_result = img
    
    return img_result

def add_noise(x:tc.Tensor,max_randn_rate=0.1,randn_rate=None,x_already_normed=False):
    """input.shape=(batch,f1,f2,...) output's var will be normalizate  """
    ndim=x.ndim-1
    if x_already_normed:
        x_std=tc.ones([x.shape[0]]+[1]*ndim,device=x.device,dtype=x.dtype)
        x_mean=tc.zeros([x.shape[0]]+[1]*ndim,device=x.device,dtype=x.dtype)
    else: 
        dim=list(range(1,x.ndim))
        x_std=x.std(dim=dim,keepdim=True)
        x_mean=x.mean(dim=dim,keepdim=True)
    if randn_rate is None:
        randn_rate=max_randn_rate*np.random.rand()*tc.rand(x_mean.shape,device=x.device,dtype=x.dtype)
    cache=(x_std**2+(x_std*randn_rate)**2)**0.5
    return (x-x_mean+tc.randn(size=x.shape,device=x.device,dtype=x.dtype)*randn_rate*x_std)/(cache+1e-7)
 
def dice_coef(y_pred:tc.Tensor,y_true:tc.Tensor, thr=0.5, dim=(-1,-2), epsilon=0.001):
    y_pred = y_pred.sigmoid()
    y_true = y_true.to(tc.float32)
    y_pred = (y_pred > thr).to(tc.float32)
    inter = (y_true * y_pred).sum(dim = dim)
    den = y_true.sum(dim = dim) + y_pred.sum(dim = dim)
    dice = ((2 * inter + epsilon)/(den + epsilon)).mean()
    return dice

# based on:
# https://github.com/kevinzakka/pytc-goodies/blob/master/losses.py


