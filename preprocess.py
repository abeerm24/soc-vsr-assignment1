# 
# This file contains helper functions necessary for processing HDF5 data
# Use it to convert the image data to a .h5 file
#
# TASKS :-
# Complete the helper functions
# 

import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
import PIL.Image as pil_image
import argparse


def convert_rgb_to_y(img):
    # Converts rgb image to y
    (n,m,channels) = img.shape
    if type(img) == np.ndarray:
        outputImage = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                (r,g,b) = tuple(img[i][j])
                outputImage[i][j] = 16 + 65.738*r/256  + 129.057*g/256 + 25.064*b/256
        return outputImage
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        outputImage = torch.zeros((n,m))
        for i in range(n):
            for j in range(m):
                (r,g,b) = tuple(img[i][j])
                outputImage[i][j] = 16 + 65.738*r/256  + 129.057*g/256 + 25.064*b/256
        return outputImage
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    # Converts an rgb image to ycbcr
    (n,m,channels) = img.shape
    if type(img) == np.ndarray:
        for i in range(n):
            for j in range(m):
                (r,g,b) = tuple(img[i][j])
                y = 16 + 65.738*r/256  + 129.057*g/256 + 25.064*b/256
                cb = 128 - 37.945*r/256 - 74.494*g/256 + 112.439*b/256
                cr = 128 + 112.539/256*r - 94.154*g/256 - 18.295*b/256 
                img[i][j] = np.array((y,cb,cr),dtype=int)
        return img 
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)

        for i in range(n):
            for j in range(m):
                (r,g,b) = tuple(img[i][j])
                y = int(16 + 65.738*r/256  + 129.057*g/256 + 25.064*b/256)
                cb = int(128 - 37.945*r/256 - 74.494*g/256 + 112.439*b/256)
                cr = int(128 + 112.539/256*r - 94.154*g/256 - 18.295*b/256) 
                img[i][j] = torch.tensor((y,cb,cr))
        return img 
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    # Converts an image from ycbcr format to rgb
    (n,m,channels) = img.shape
    if type(img) == np.ndarray:
        for i in range(n):
            for j in range(m):
                (y,cb,cr) = tuple(img[i][j])
                r = y + 1.402*cr
                g = y - 0.344136*cb - 0.714136*cr
                b = y + 1.772*cb
                img[i][j] = np.array((r,g,b),dtype=int)
        return img

    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        img = torch.tensor(img)
        for i in range(n):
            for j in range(m):
                (y,cb,cr) = tuple(img[i][j])
                r = int(y + 1.402*cr)
                g = int(y - 0.344136*cb - 0.714136*cr)
                b = int(y + 1.772*cb)
                img[i][j] = torch.tensor((r,g,b))
        return img
        
    else:
        raise Exception('Unknown Type', type(img))


def preprocess(args):

    # args is an object returned by an argument parser
    # Required attributes and sample values are provided
    # --output_path path/
    # --images_dir path/
    # --stride 1 
    # --patch size 4 
    # --scale 2  

    print('Preprocessing started')

    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        print(image_path)  #Added for debugging
        dataFile = h5py.File(image_path, 'r')
            # hr = pil_image.open(image_path).convert('RGB')
            # hr_width = (hr.width // args.scale) * args.scale
            # hr_height = (hr.height // args.scale) * args.scale
            # hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            # lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
            # lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            # hr = np.array(hr).astype(np.float32)
            # lr = np.array(lr).astype(np.float32)
            # hr = convert_rgb_to_y(hr)
            # lr = convert_rgb_to_y(lr)
        hrImages = dataFile.get('hr')
        lrImages = dataFile.get('lr')

        for x in range(hrImages.shape[0]):
            hr = hrImages[x]
            lr = lrImages[x]
            for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
                for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                    lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                    hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])
    
        print("Preprocessing complete") #Added for debugging
        break

    print("Converting to numpy arrays") #Added for debugging
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    print("Creating dataset")           #Added for debugging
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    print("Saving file")  #Added for debugging
    h5_file.close()
    print("File saved and closed") #Added for debugging

#Initialize parser
parser = argparse.ArgumentParser("Takes in input about output path, image dir path, stride, patch size, scale")

#Add arguments to parser
parser.add_argument('--output_path',help="location of h5 file where image patches will be stored")
parser.add_argument('--images_dir',help='Location of h5 files with stored images')
parser.add_argument('--stride',type=int,help='Stride size (int)')
parser.add_argument('--patch_size',type=int,help='Patch size (int)')
parser.add_argument('--scale', type=int, help='Scaling factor (int)')

#Collect args from commandline
args = parser.parse_args()
print("Process complete") #Added for debugging

#Execute preprocessing
preprocess(args)
