# Use this to test your model. 
# Use appropriate command line arguments and conditions

import argparse
# from model import Net
import model1
import torch
import glob
import PIL.Image as pil_image
import numpy as np
import matplotlib.pyplot as plt
from metrics import calc_psnr

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


# Code for the 8x8 patch model

# loaded_model = Net()
# FILE = "nn-model-8x8-16b.pth"
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()

#Code for 4x4 patch model

loaded_model = model1.Net()
FILE = "nn-model9.pth"
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()


parser = argparse.ArgumentParser()

parser.add_argument("--images_dir", help="Location of test images")
parser.add_argument("--patch_size", help="Patch size",type=int)
parser.add_argument("--stride", help="Stride for patches", type=int)

args = parser.parse_args()

# create figure
fig = plt.figure(figsize=(15, 15))
rows = 8 
cols = 2

ctrl = 1
print("Running started")
for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
    inputImg = pil_image.open(image_path).convert('RGB')
    inputTitle = "Input image: " + image_path
    
    inputImg = np.array(inputImg)

    pil_image.fromarray(inputImg).show(inputTitle)

    # fig.add_subplot(rows, cols, ctrl)
    # plt.imshow(np.array(inputImg))
    # plt.axis('off')
    # plt.title(inputTitle)
    
    #Generate patches from the yChannel

    (n,m,c) = inputImg.shape

    lr_patches = []
    for i in range(0,n - args.patch_size + 1, args.stride):  #Patch size = 4, stride = 1
        for j in range(0,m - args.patch_size + 1, args.stride):
            lr_patches.append(inputImg[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = torch.tensor(np.array(lr_patches), dtype= torch.float)
    
    hr_patches = loaded_model(lr_patches)
    (N,n,m,c) = lr_patches.shape
    print(N)
    hr_patches = hr_patches.view(N,n,m,c)

    hr_patches = hr_patches.detach().numpy()

    # print("Loss 1 ", hr_patches[0]-np.array(lr_patches[0]))
    # print(hr_patches[0]-np.array(lr_patches[0]))

    #Reconstructing image from high resolution patches

    (n,m,c) = inputImg.shape

    newImg = np.zeros((n,m,c+1))

    for patch_num in range(N):
        x = patch_num % (n - args.patch_size + 1) 
        y = int(patch_num / (n - args.patch_size + 1)) 
        for i in range(args.patch_size):
            for j in range(args.patch_size):
                for channel in range(3):
                    if y+i<n and x+j < m: newImg[y+i][x+j][channel] += hr_patches[patch_num][i][j][channel]
                if y+i<n and x+j < m: newImg[y+i][x+j][3] += 1
    
    for i in range(n):
        for j in range(m):
            for channel in range(3):
                newImg[i][j][channel] /= newImg[i][j][3]
    
    #yChannelNew = np.array(yChannelNew, dtype=np.int32)

    # ycbcrImgNew = ycbcrImg
    # ycbcrImgNew[:,:,0] = yChannelNew[0]

    # rgbImg = convert_ycbcr_to_rgb(ycbcrImgNew)

    rgbImg = newImg[:,:,0:3]
    rgbImg = np.array(rgbImg, dtype=np.uint8)

    for i in range(rgbImg.shape[0]):
        for j in range(rgbImg.shape[1]):
            for k in range(rgbImg.shape[2]):
                if (rgbImg[i][j][k]>255): rgbImg[i][j][k] = 255
                if (rgbImg[i][j][k]<0): rgbImg[i][j][k] = 0

    outputImg = pil_image.fromarray(rgbImg)
    outputTitle = "Output image: " + image_path
    # print("Losses:")
    # print(rgbImg - inputImg)
    outputImg.show(outputTitle)

    fileLocation = "image" + str(ctrl) + ".png"

    print("Output image shape: ", rgbImg.shape)
    print("Input image shape", inputImg.shape)

    #Print the PSNR
    print("Peak signal to noise ratio (in dB): ", calc_psnr(rgbImg,inputImg))

    # outputImg.save(fileLocation)
    # fig.add_subplot(rows, cols, ctrl)
    # plt.imshow(rgbImg)
    # plt.axis('off')
    # plt.title(outputTitle)

    ctrl+=1
