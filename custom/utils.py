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

def windowing_thorax(img_png, npy, channel=3):
    dcm = npy.copy()
    img_rows = 512
    img_cols = 512

    if channel == 1:
        npy = npy.squeeze()
        npy = cv2.resize(npy, (512,512), interpolation = cv2.INTER_LINEAR)
        npy = npy + 40 ## change to lung/med setting
        npy = np.clip(npy, 0, 160)
        npy = npy / 160
        npy = 255 * npy
        npy = npy.astype(np.uint8)

    elif channel == 3:
        dcm1 = dcm[0] + 150
        dcm1 = np.clip(dcm1, 0, 400)
        dcm1 = dcm1 / 400.
        dcm1 *= (2**8-1)
        dcm1 = dcm1.astype(np.uint8) 
    	
        dcm2 = dcm[0] - 250
        dcm2 = np.clip(dcm2, 0, 100)
        dcm2 = dcm2 / 100.
        dcm2 *= (2**8-1)
        dcm2 = dcm2.astype(np.uint8) 
    	
        dcm3 = dcm[0] + 950 
        dcm3 = np.clip(dcm3, 0, 1000)
        dcm3 = dcm3 / 1000.
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

#----------------------------------------------------------------------------

# Image utils to save figures

def smooth_image(image, threshold=7.5e4, kernel_x=6):
    kernel = np.ones((kernel_x,kernel_x),np.float32)/(kernel_x**2)
    return cv2.filter2D(image,-1,kernel)

def convert_to_numpy_array(image, drange=[0,1], rgbtogray=False):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    
    if rgbtogray:
        return convert_rgb_to_gray(image)
    
    return image

def convert_to_pil_image(image, drange=[0,1]):
    import PIL.Image
    
    image = convert_to_numpy_array(image, drange)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, fmt)

def convert_rgb_to_gray(image):
#     return np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def create_summary_figure(img_in_np, img_out_np, mask, i, checkpoint=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    img_in_np_tmp = adjust_dynamic_range(img_in_np, [0,255], [-1,1])
    img_out_np_tmp = adjust_dynamic_range(img_out_np, [0,255], [-1,1])
    img_diff = img_in_np_tmp - img_out_np_tmp
    img_diff_smooth = smooth_image(img_diff)
    img_diff_std = img_diff_smooth.std()

    fig=plt.figure(figsize=(16, 4))
    fig.add_subplot(1, 4, 1)
    plt.title('Input (real)')
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(img_in_np_tmp, cmap=plt.cm.gray) #transpose(img_in_np))

    if (len(mask) > 0):
        mask_temp = np.flip(mask[i], 0)
        mask_temp = np.ma.masked_where(mask_temp == 0, mask_temp)
        plt.imshow(mask_temp, alpha=0.7, cmap=plt.cm.autumn)

    fig.add_subplot(1, 4, 2)
    plt.title('Output (fake)') if checkpoint else plt.title('Output (fake) iter: {}'.format(checkpoint))
    plt.axis('off')
    plt.imshow(img_out_np_tmp, cmap=plt.cm.gray)

    fig.add_subplot(1, 4, 3)
    plt.title('Difference (+)')
    plt.axis('off')
    plt.imshow(img_in_np_tmp, cmap=plt.cm.gray)

    norm = mpl.colors.Normalize(vmin=0, vmax=0.2)
    img_diff_smooth[(img_diff_smooth < img_diff_std*0.5) & (img_diff_smooth > img_diff_std*-0.5)] = 0.
    plt.imshow(img_diff_smooth, cmap='inferno', alpha=0.4, norm=norm)

    fig.add_subplot(1, 4, 4)
    plt.title('Difference (-)')
    plt.axis('off')
    plt.imshow(img_in_np_tmp, cmap=plt.cm.gray)

    vstd = img_diff_std

#     norm = mpl.colors.Normalize(vmin=0, vmax=vstd*5)
    img_diff_smooth[(img_diff_smooth < img_diff_std*0.5) & (img_diff_smooth > img_diff_std*-0.5)] = 0.
    plt.imshow(img_diff_smooth*-1, cmap='inferno', alpha=0.4, norm=norm)
  
    return fig
    

def apply_mirror_augment(minibatch):
    mask = np.random.rand(minibatch.shape[0]) < 0.5
    minibatch = np.array(minibatch)
    minibatch[mask] = minibatch[mask, :, :, ::-1]
    return minibatch

#----------------------------------------------------------------------------
