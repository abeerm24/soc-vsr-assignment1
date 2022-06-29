# This file will store functions that will present a performance metric of whatever image processing is performed
#
#
# TASKS:-
# Implement metrics as required

import torch
import numpy as np
from math import log10, sqrt

def calc_psnr(img1, img2):
    #Calculate mean squarred error
    mse = np.mean((img1 - img2) ** 2)
    max_pixel = 255 #Maximum pixel level value for RGB image is 255
    psnr = 20*log10(max_pixel/sqrt(mse))
    return psnr