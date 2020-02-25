# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""utility functions."""

import os
import cv2
import numpy as np
import PIL.Image
import PIL.ImageFont

#----------------------------------------------------------------------------

# Image utils.

def windowing_brain(img_png, npy, channel=3): 
    dcm = npy.copy()
    img_rows = 512
    img_cols = 512
    
    if channel == 1:
        npy = npy.squeeze()
        npy = cv2.resize(npy, (512,512), interpolation = cv2.INTER_LINEAR)
        npy = npy + 40
        npy = np.clip(npy, 0, 160)
        npy = npy / 160
        npy = 255 * npy
        npy = npy.astype(np.uint8)
        
    elif channel == 3:
        dcm1 = dcm[0] + 0
        dcm1 = np.clip(dcm1, 0, 80)
        dcm1 = dcm1 / 80.
        dcm1 *= (2**8-1)
        dcm1 = dcm1.astype(np.uint8)
        
        dcm2 = dcm[0] + 20
        dcm2 = np.clip(dcm2, 0, 200)
        dcm2 = dcm2 / 200.
        dcm2 *= (2**8-1)
        dcm2 = dcm2.astype(np.uint8)
        
        dcm3 = dcm[0] - 5
        dcm3 = np.clip(dcm3, 0, 50)
        dcm3 = dcm3 / 50.
        dcm3 *= (2**8-1)
        dcm3 = dcm3.astype(np.uint8)
        
        npy = np.zeros([img_rows,img_cols,3], dtype=int)
        npy[:,:,0] = dcm2
        npy[:,:,1] = dcm1
        npy[:,:,2] = dcm3
        
    return npy

def write_png_image(img_png, npy):
    if not os.path.exists(img_png):
        return cv2.imwrite(img_png, npy)
    else:
        return False
    

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def convert_to_pil_image(image, drange=[0,1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, fmt)

def convert_rgb_to_gray(image):
    return np.array([np.dot(image[i][...,:3], [0.299, 0.587, 0.114]) for i in range(3)])

def save_image_grid(images, filename, drange=[0,1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)

def apply_mirror_augment(minibatch):
    mask = np.random.rand(minibatch.shape[0]) < 0.5
    minibatch = np.array(minibatch)
    minibatch[mask] = minibatch[mask, :, :, ::-1]
    return minibatch

#----------------------------------------------------------------------------
