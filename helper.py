import torch as tc
import numpy as np

# Some helper functions for inference and training

def to_original(im_after, img, image_size=1024):
    """
    Resizes the image back to its original size.

    Args:
        im_after (numpy.ndarray): Resized image.
        img (numpy.ndarray): Original image.
        image_size (int): Target size of the image.

    Returns:
        numpy.ndarray: Image resized to its original size.
    """
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

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    Args:
        mask (numpy.ndarray): Binary mask to encode.

    Returns:
        str: Encoded binary mask.
    """
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

def min_max_normalization(x: tc.Tensor) -> tc.Tensor:
    """
    Performs min-max normalization of a PyTorch tensor.

    Args:
        x (torch.Tensor): Input tensor to normalize.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    shape = x.shape
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    min_ = x.min(dim=-1, keepdim=True)[0]
    max_ = x.max(dim=-1, keepdim=True)[0]
    if min_.mean() == 0 and max_.mean() == 1:
        return x.reshape(shape)

    x = (x - min_) / (max_ - min_ + 1e-9)
    return x.reshape(shape)

def norm_with_clip(x: tc.Tensor, smooth=1e-5):
    """
    Normalizes a PyTorch tensor with clipping.

    Args:
        x (torch.Tensor): Input tensor to normalize.
        smooth (float): Smoothing factor.

    Returns:
        torch.Tensor: Normalized tensor with clipping.
    """
    dim = list(range(1, x.ndim))
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + smooth)
    x[x > 5] = (x[x > 5] - 5) * 1e-3 + 5
    x[x < -3] = (x[x < -3] + 3) * 1e-3 - 3
    return x

def add_edge(x: tc.Tensor, edge: int):
    """
    Adds edges to an image tensor.

    Args:
    - x (tc.Tensor): Input tensor of shape (C, H, W), where C is the number of channels,
                     H is the height, and W is the width.
    - edge (int): Number of edge pixels to add to each side of the image.

    Returns:
    - tc.Tensor: Output tensor with edges added, shape (C, H+2*edge, W+2*edge).
    """
    mean_ = int(x.to(tc.float32).mean())
    x = tc.cat([x, tc.ones([x.shape[0], edge, x.shape[2]], dtype=x.dtype, device=x.device) * mean_], dim=1)
    x = tc.cat([x, tc.ones([x.shape[0], x.shape[1], edge], dtype=x.dtype, device=x.device) * mean_], dim=2)
    x = tc.cat([tc.ones([x.shape[0], edge, x.shape[2]], dtype=x.dtype, device=x.device) * mean_, x], dim=1)
    x = tc.cat([tc.ones([x.shape[0], x.shape[1], edge], dtype=x.dtype, device=x.device) * mean_, x], dim=2)
    return x

def to_1024_1024(img, CFG, image_size=1024):
    """
    Resizes the image to 1024x1024 with rotation if necessary.

    Args:
    - img (np.ndarray): Input image.
    - CFG: Configuration object.
    - image_size (int): Desired size of the image.

    Returns:
    - np.ndarray: Resized image.
    """
    if image_size > img.shape[1]:
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

def add_noise(x:tc.Tensor, max_randn_rate=0.1, randn_rate=None, x_already_normed=False):
    """
    Adds random noise to the input tensor.

    Args:
    - x (tc.Tensor): Input tensor.
    - max_randn_rate (float): Maximum rate of random noise.
    - randn_rate: Rate of random noise. If None, it's randomly generated.
    - x_already_normed (bool): Indicates whether the input tensor is already normalized.

    Returns:
    - tc.Tensor: Output tensor with added noise.
    """
    # The number of dimensions excluding batch dimension
    ndim = x.ndim-1

    if x_already_normed:
        x_std = tc.ones([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
        x_mean = tc.zeros([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
    else: 
        dim = list(range(1, x.ndim))
        x_std = x.std(dim=dim, keepdim=True)
        x_mean = x.mean(dim=dim, keepdim=True)

    # Random noise
    if randn_rate is None:
        randn_rate = max_randn_rate * np.random.rand()*tc.rand(x_mean.shape,device=x.device,dtype=x.dtype)
    
    # Calc noise based on mean and sd.
    cache=(x_std**2+(x_std*randn_rate)**2)**0.5
    noise = tc.randn(size=x.shape, device=x.device, dtype=x.dtype) * randn_rate * x_std
    return (x - x_mean + noise) / (cache + 1e-7)

def dice_coef(y_pred:tc.Tensor, y_true:tc.Tensor, thr=0.5, dim=(-1,-2), epsilon=0.001):
    """
    Computes the Dice coefficient between predicted and true binary tensors.

    Args:
    - y_pred (tc.Tensor): Predicted tensor.
    - y_true (tc.Tensor): Ground truth tensor.
    - thr (float): Threshold for binary conversion.
    - dim (tuple): Dimensions along which to compute Dice coefficient.
    - epsilon (float): Small value for numerical stability.

    Returns:
    - tc.Tensor: Dice coefficient.
    """
    y_pred = y_pred.sigmoid()

    y_true = y_true.to(tc.float32)
    y_pred = (y_pred > thr).to(tc.float32)

    inter = (y_true * y_pred).sum(dim = dim)
    den = y_true.sum(dim = dim) + y_pred.sum(dim = dim)

    dice = ((2 * inter + epsilon)/(den + epsilon)).mean()
    return dice