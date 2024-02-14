import torch as tc
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F

from .one_hot import one_hot

# Function to resize image to 1024x1024 without rotation
def to_1024_no_rot(img, image_size=1024):
    if image_size > img.shape[0]:
        # Calculate padding for top and bottom
        start1 = (image_size - img.shape[0]) // 2
        top = img[0: start1, 0: img.shape[1]]
        bottom = img[img.shape[0] - start1: img.shape[0], 0: img.shape[1]]
        
        # Concatenate top, image, and bottom
        img_result = np.concatenate((top, img, bottom), axis=0)
    else:
        img_result = img
    
    return img_result

# Function to resize image to 1024x1024 using to_1024 function
def to_1024_1024(img, image_size=1024):
    img_result = to_1024(img, image_size)
    return img_result

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
    """input.shape=(batch,f1,...)"""
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
def to_1024(img, image_size=1024):
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
 

# based on:
# https://github.com/kevinzakka/pytc-goodies/blob/master/losses.py

class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = tc.randn(1, N, 3, 5, requires_grad=True)
        >>> target = tc.empty(1, 3, 5, dtype=tc.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: tc.Tensor,
            target: tc.Tensor) -> tc.Tensor:
        if not tc.is_tensor(input):
            raise TypeError("Input type is not a tc.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = tc.sum(input_soft * target_one_hot, dims)
        cardinality = tc.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return tc.mean(1. - dice_score)

def dice_loss(
        input: tc.Tensor,
        target: tc.Tensor) -> tc.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~tcgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss()(input, target)